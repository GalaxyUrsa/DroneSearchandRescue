import numpy as np
from math import sqrt
from cv2 import line
import time
import sys
import os
def draw_circle(image, center, radius, thickness=2):
    """
    Args:
        image: 输入图片
        center: 圆的中心坐标
        radius: 圆的半径
        thickness: 线的粗细
    """
    R = radius * 1.2
    rows, cols = image.shape[:2]
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.logical_and(dist >= R - thickness, dist <= R + thickness)
    image[mask] = (255, 0, 0)  # 给区域内的像素赋予红色color
    
    
def draw_rect(image, xyxy, thickness=1, corner_size=7, color=(0, 0, 255)):
    """
    Args:
        image: 输入图片
        xyxy: 检测框坐标(x1, y1, x2, y2) x1y1是检测框左上角点，x2y2是检测框右下角点
        thickness: 线的粗细
        corner_size: 检测框四个角的线的粗细
        color: 框的颜色

    """
    # print(color)
    x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    # 左上角
    line(image, (x, y), (x + corner_size, y), color, thickness)
    line(image, (x, y), (x, y + corner_size), color, thickness)
    # 右上角
    line(image, (x + w, y), (x + w - corner_size, y), color, thickness)
    line(image, (x + w, y), (x + w, y + corner_size), color, thickness)
    # 左下角
    line(image, (x, y + h), (x + corner_size, y + h), color, thickness)
    line(image, (x, y + h), (x, y + h - corner_size), color, thickness)
    # 右下角
    line(image, (x + w, y + h), (x + w - corner_size, y + h), color, thickness)
    line(image, (x + w, y + h), (x + w, y + h - corner_size), color, thickness)