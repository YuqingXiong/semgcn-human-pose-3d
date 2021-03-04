from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from models.sem_gcn import SemGCN


def parse_args():
    """
    命令行参数使用
    """
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments / 常规参数
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')

    # Model arguments   / 模型参数
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--non_local', dest='non_local', action='store_true', help='if use non-local layers')
    parser.set_defaults(non_local=False)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')

    # Experimental  /实验参数
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    args = parser.parse_args()

    # Check invalid configuration   / 检查无效的配置
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')    # 加载数据，默认是h36m
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)     # ===========================================dataset的初始由来
        subjects_train = TRAIN_SUBJECTS     # ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS       # ['S9', 'S11']
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)     # dataset的改变，其实函数里没变

    print('==> Loading 2D detections...')
    # ================================================================== input_2d的原始来源：keypoints[subject][action]
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cpu")      # 原始为cuda

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCN(adj, args.hid_dim, num_layers=args.num_layers, p_dropout=p_dropout,
                       nodes_group=dataset.skeleton().joints_group() if args.non_local else None).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)     # 损失函数
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i] = evaluate(valid_loader, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        exit(0)
    # ========================================================
    # poses_2d的fetch操作，poses_train_2d是从keypoints[subject][action]中得来的
    poses_train, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)
    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return


def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        """
        inputs_2d : 模型的输入参数 =========================================================================================《《《
        targets_3d : 目标结果
        """
        # print(inputs_2d[0])
        # print(inputs_2d.shape)
        outputs_3d = model_pos(inputs_2d)   # inputs_2d到model中得到outputs_3d============================================

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d).cpu()
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())