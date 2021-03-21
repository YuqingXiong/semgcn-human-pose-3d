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
    dataset_path = "./data/data_3d_h36m.npz"    # 加载数据
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
    ckpt_path = "./checkpoint/pretrained/ckpt_semgcn_nonlocal_sh.pth.tar"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_pos.load_state_dict(ckpt['state_dict'], False)
    model_pos.eval()
    # ============ 新增代码 ==============
    # 从项目处理2d数据的代码中输出的一个人体数据
    inputs_2d = [[483.0, 450], [503, 450], [503, 539], [496, 622], [469, 450], [462, 546], [469, 622], [483, 347],
                 [483, 326], [489, 264], [448, 347], [448, 408], [441, 463], [517, 347], [524, 408], [538, 463]]

    # # openpose的测试样例识别结果
    # inputs_2d = [[86.0, 137], [99, 128], [94, 127], [97, 110], [89, 105], [102, 129], [116, 116], [99, 110],
    #              [105, 93], [117, 69], [147, 63], [104, 93], [89, 69], [82, 38], [89, 139], [94, 140]]

    inputs_2d = np.array(inputs_2d)
    # inputs_2d[:, 1] = np.max(inputs_2d[:, 1]) - inputs_2d[:, 1]   # 变成正的人体姿态，原始数据为倒立的

    cam = dataset.cameras()['S1'][0]    # 获取相机参数
    inputs_2d[..., :2] = normalize_screen_coordinates(inputs_2d[..., :2], w=cam['res_w'], h=cam['res_h'])  # 2d坐标处理

    # 画出归一化屏幕坐标并且标记序号的二维关键点图像
    print(inputs_2d)    # 打印归一化后2d关键点坐标
    d_x = inputs_2d[:, 0]
    d_y = inputs_2d[:, 1]
    plt.figure()
    plt.scatter(d_x, d_y)
    for i, txt in enumerate(np.arange(inputs_2d.shape[0])):
        plt.annotate(txt, (d_x[i], d_y[i]))     # 标号
    # plt.show()      # 显示2d关键点归一化后的图像

    # 获取3d结果
    inputs_2d = torch.tensor(inputs_2d, dtype=torch.float32)    # 转换为张量
    outputs_3d = model_pos(inputs_2d).cpu()         # 加载模型
    outputs_3d[:, :, :] -= outputs_3d[:, :1, :]     # Remove global offset / 移除全球偏移
    predictions = [outputs_3d.detach().numpy()]     # 预测结果
    prediction = np.concatenate(predictions)[0]     # 累加取第一个
    # Invert camera transformation  / 反相机的转换
    prediction = camera_to_world(prediction, R=cam['orientation'], t=0)     # R和t的参数设置影响不大，有多种写法和选取的相机参数有关，有些S没有t等等问题
    prediction[:, 2] -= np.min(prediction[:, 2])    # 向上偏移min(prediction[:, 2]),作用是把坐标变为正数
    print('prediction')
    print(prediction)   # 打印画图的3d坐标
    plt.figure()
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    o_x = prediction[:, 0]
    o_y = prediction[:, 1]
    o_z = prediction[:, 2]
    print(o_x)
    print(o_y)
    print(o_z)
    ax.scatter(o_x, o_y, o_z)

    temp = o_x
    x = [temp[9], temp[8], temp[7], temp[10], temp[11], temp[12]]
    temp = o_y
    y = [temp[9], temp[8], temp[7], temp[10], temp[11], temp[12]]
    temp = o_z
    z = [temp[9], temp[8], temp[7], temp[10], temp[11], temp[12]]
    ax.plot(x, y, z)

    temp = o_x
    x = [temp[7], temp[0], temp[4], temp[5], temp[6]]
    temp = o_y
    y = [temp[7], temp[0], temp[4], temp[5], temp[6]]
    temp = o_z
    z = [temp[7], temp[0], temp[4], temp[5], temp[6]]
    ax.plot(x, y, z)

    temp = o_x
    x = [temp[0], temp[1], temp[2], temp[3]]
    temp = o_y
    y = [temp[0], temp[1], temp[2], temp[3]]
    temp = o_z
    z = [temp[0], temp[1], temp[2], temp[3]]
    ax.plot(x, y, z)

    temp = o_x
    x = [temp[7], temp[13], temp[14], temp[15]]
    temp = o_y
    y = [temp[7], temp[13], temp[14], temp[15]]
    temp = o_z
    z = [temp[7], temp[13], temp[14], temp[15]]
    ax.plot(x, y, z)

    # temp = o_x
    # x = [temp[0], temp[14]]
    # temp = o_y
    # y = [temp[0], temp[14]]
    # temp = o_z
    # z = [temp[0], temp[14]]
    # ax.plot(y, x, z)
    #
    # temp = o_x
    # x = [temp[0], temp[15]]
    # temp = o_y
    # y = [temp[0], temp[15]]
    # temp = o_z
    # z = [temp[0], temp[15]]
    # ax.plot(y, x, z)

    # 改变坐标比例的代码，该代码的效果是z坐标轴是其他坐标的两倍
    from matplotlib.pyplot import MultipleLocatort
    major_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.zaxis.set_major_locator(major_locator)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1]))

    plt.show()


if __name__ == '__main__':
    main()