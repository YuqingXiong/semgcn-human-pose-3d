from __future__ import print_function, absolute_import, division

import time
import argparse
import numpy as np
import os.path as path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.utils import AverageMeter
from common.data_utils import read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from common.camera import camera_to_world, image_coordinates
from common.visualization import render_animation
from common.h36m_dataset import Human36mDataset
#
# dataset_path = path.join('data', 'data_3d_' + 'h36m' + '.npz')
# dataset = Human36mDataset(dataset_path)
# # Invert camera transformation  / 反相机的转换
# cam = dataset.cameras()['S11'][0]
# prediction = camera_to_world(prediction, R=cam['orientation'], t=0)
# prediction[:, :, 2] -= np.min(prediction[:, :, 2])

from common.camera import world_to_camera, normalize_screen_coordinates
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')


outputs_3d = [[534.7947, 308.0375, -791.3116],
             [591.6291, 151.0018, -1016.9899],
             [393.3173, 338.0340, -1155.9219],
             [366.4056, 421.9557, -1100.2411],
             [391.8182, 280.0039, -1027.4448],
             [122.2689, 354.1458, -1547.6687],
             [161.8528, 328.8372, -1315.0308],
             [464.0311, 143.1224, -887.7397],
             [275.7003, 188.9405, -758.0136],
             [532.3159, -33.3475, -1206.9071],
             [416.1473, 60.8146, -768.9236],
             [169.8227, 270.8355, -566.0059],
             [222.3016, 208.2446, -557.9625],
             [537.9346, 69.7744, -830.5811],
             [353.7056, 347.3874, -688.0934],
             [574.3894, 232.4201, -673.0931]]

# outputs_3d = torch.tensor(outputs_3d, dtype=torch.float32)
# outputs_3d[:, :] -= outputs_3d[:, :1]
# dataset_path = path.join('data', 'data_3d_' + 'h36m' + '.npz')
# dataset = Human36mDataset(dataset_path)
# # Invert camera transformation  / 反相机的转换
# cam = dataset.cameras()['S11'][0]
# outputs_3d = camera_to_world(outputs_3d, R=cam['orientation'], t=0)
# outputs_3d[:, :] -= np.min(outputs_3d[:, :])
input_2d = [[483.0, 450], [503, 450], [503, 539], [496, 622], [469, 450], [462, 546], [469, 622], [483, 347],
            [483, 326], [489, 264], [448, 347], [448, 408], [441, 463], [517, 347], [524, 408], [538, 463], [0,0]]
input_2d = np.array(input_2d)

d_x = input_2d[:,0]
d_y = input_2d[:,1]
plt.scatter(d_x, 630 - d_y)

outputs_3d = [[-0.3306, -0.9054, -0.1337],
         [-0.2866, -1.0783, -0.5198],
         [-0.1885, -1.5564, -0.9731],
         [-0.0227, -1.7251, -1.0269],
         [-0.3604, -1.0299, -0.4171],
         [-0.3347, -1.7848, -0.6212],
         [ 0.0748, -2.1566, -1.2846],
         [-0.2870, -0.3238, -0.3251],
         [-0.2700, -0.6638, -0.6708],
         [-0.1797, -0.2387, -1.0865],
         [-0.1934, -0.4400, -0.7992],
         [-0.1941, -1.0662, -0.4102],
         [-0.1707, -1.3335, -0.7349],
         [-0.1214, -0.4815, -0.8129],
         [-0.1152, -0.9897, -0.7091],
         [ 0.0649, -1.1301, -0.8009]]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

outputs_3d = np.array(outputs_3d)
o_x = outputs_3d[:, 0]
o_y = outputs_3d[:, 1]
o_z = outputs_3d[:, 2]
ax.scatter(o_x, o_y, o_z)

plt.show()