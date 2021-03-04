# import torch
# import numpy as np
# poses_2d = np.array([[483.0,450],[503, 450],[503, 539],[496, 622],[469, 546], [462, 546.],[469, 622],[483, 347],
#                      [483, 326],[489, 264],[448, 347],[448, 408],[441, 463],[517, 347],[524, 408],[538, 463]])
# model = torch.load("checkpoint/pretrained/ckpt_semgcn_nonlocal_sh.pth.tar", map_location='cpu')
# poses_3d = model(poses_2d)
# print(poses_3d)

from __future__ import print_function, absolute_import, division

import numpy as np
import os.path as path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from common.data_utils import read_3d_data, create_2d_data
from common.camera import camera_to_world, image_coordinates
from common.camera import world_to_camera, normalize_screen_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

def main():
    dataset_path = "./data/data_3d_h36m.npz"
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
    dataset = read_3d_data(dataset)
    cudnn.benchmark = True
    device = torch.device("cpu")
    from models.sem_gcn import SemGCN
    from common.graph_utils import adj_mx_from_skeleton
    p_dropout = None
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCN(adj, 128, num_layers=4, p_dropout=p_dropout,
                       nodes_group=dataset.skeleton().joints_group()).to(device)
    ckpt_path = "./checkpoint/pretrained/ckpt_semgcn_nonlocal.pth.tar"
    ckpt = torch.load(ckpt_path,map_location='cpu')
    start_epoch = ckpt['epoch']
    error_best = ckpt['error']
    model_pos.load_state_dict(ckpt['state_dict'], False)
    model_pos.eval()

    input_2d = [[483.0, 450], [503, 450], [503, 539], [496, 622], [469, 450], [462, 546], [469, 622], [483, 347],
                [483, 326], [489, 264], [448, 347], [448, 408], [441, 463], [517, 347], [524, 408], [538, 463]]
    input_2d = np.array(input_2d)
    input_2d[:, 1] = 630 - input_2d[:, 1]
    # # ============ 新增代码 ==============
    cam = dataset.cameras()['S1'][0]
    input_2d[..., :2] = normalize_screen_coordinates(input_2d[..., :2], w=cam['res_w'], h=cam['res_h'])  # 2d坐标处理
    print(input_2d)
    d_x = input_2d[:, 0]
    d_y = input_2d[:, 1]
    plt.scatter(d_x, d_y)
    plt.show()      # 显示2d关键点归一化后的图像

    # 获取3d结果
    input_2d = torch.tensor(input_2d, dtype=torch.float32)
    outputs_3d = model_pos(input_2d).cpu()
    print(outputs_3d)
    outputs_3d[:, :, :] -= outputs_3d[:, :1, :]      # Remove global offset
    outputs_3d = outputs_3d.detach().numpy()
    print(outputs_3d)

    outputs_3d = np.concatenate(outputs_3d)

    # Invert camera transformation  / 反相机的转换
    outputs_3d = camera_to_world(outputs_3d, R=cam['orientation'], t=0)
    print('2')
    print(outputs_3d)
    outputs_3d[:, 2] -= np.min(outputs_3d[:, 2])  # 原始有三维
    print(outputs_3d)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    o_x = outputs_3d[:, 0]
    o_y = outputs_3d[:, 1]
    o_z = outputs_3d[:, 2]
    print(o_x)
    print(o_y)
    print(o_z)
    ax.scatter(o_x, o_y, o_z)

    from matplotlib.pyplot import MultipleLocator
    major_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.zaxis.set_major_locator(major_locator)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1]))
    plt.show()


if __name__ == '__main__':
    main()