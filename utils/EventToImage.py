import cv2
import numpy as np


def voxel_to_gray(voxel:np.ndarray) -> np.ndarray:
    """
    :param voxel: (B,H,W) or (H,W)
    :returns: (H,W)
     - [0,255] -> [white,black]
    """
    # 聚集体素数据到单通道
    if len(voxel.shape) == 3:
        bins = np.shape(voxel)[0]
        voxel_sum = voxel[0]
        for i in range(1, bins):
            voxel_sum += voxel[i]
    else:
        voxel_sum = voxel

    # 生成灰度图（忽略极性）
    gray = np.absolute(voxel_sum) > 0
    gray = (gray * 255).astype(np.uint8)
    gray = (255 - gray)

    return gray


def voxel_to_rgb(voxel:np.ndarray) -> np.ndarray:
    """
    :param voxel: (B,H,W) or (H,W)
    :returns: [H,W,(R,G,B)]
    """
    # 聚集体素数据到单通道
    if len(voxel.shape) == 3:
        bins = np.shape(voxel)[0]
        voxel_sum = voxel[0]
        for i in range(1, bins):
            voxel_sum += voxel[i]
    else:
        voxel_sum = voxel

    # 转化为HSV格式图像
    hsv = np.zeros([voxel_sum.shape[0], voxel_sum.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = (voxel_sum < 0).astype(np.uint8) * 120  # 色调  120=Blue(Neg)   0=Red(Pos) 
    hsv[:, :, 1] = (np.abs(voxel_sum) > 0) * 255  # 饱和度
    hsv[:, :, 2] = 255  # 明度

    # HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).transpose((2,0,1))

    return rgb


def events_to_gray(events:list, height:int, width:int) -> np.ndarray:
    """
    :param events: a [N,4] array with each row in the form of [x, y, timestamp, polarity]
    :returns gray: [H,W]
    """
    # Inital voxel
    voxel_grid = np.zeros((height, width), np.float32)
    voxel_grid = voxel_grid.ravel()  # stretch to one-dimensional array

    # Extract information
    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)

    # Assign events to coordinates
    np.add.at(voxel_grid, xs + ys * width, 1)
    voxel_grid = np.reshape(voxel_grid, (height, width))
    
    # Trans to gray
    gray = voxel_grid > 0
    gray = (gray * 255).astype(np.uint8)
    gray = (255 - gray)

    return gray


def events_to_rgb(events:list, height:int, width:int) -> np.ndarray:
    """
    :param events: a [N,4] array with each row in the form of [x, y, timestamp, polarity]
    :returns: [H,W,(R,G,B)]
    """

    # Inital voxel
    voxel_grid = np.zeros((height, width), np.float32)
    voxel_grid = voxel_grid.ravel()  # stretch to one-dimensional array

    # Extract information
    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)
    pols = events[:, 3]

    # Assign events to coordinates
    np.add.at(voxel_grid, xs + ys * width, pols)
    voxel_grid = np.reshape(voxel_grid, (height, width))
    
    # Trans to HSV
    hsv = np.zeros([voxel_grid.shape[0], voxel_grid.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = (voxel_grid < 0).astype(np.uint8) * 120  # 色调  120=Blue(Neg)   0=Red(Pos) 
    hsv[:, :, 1] = (np.abs(voxel_grid) > 0) * 255  # 饱和度
    hsv[:, :, 2] = 255  # 明度

    # HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb
