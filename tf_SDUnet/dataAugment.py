#!/usr/bin/python
# -*- coding: utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 亮度变换 bright
   3. 对比度变换 contrast
   4. 随机修剪 random crop
   5. PCA抖动 PCA jittering
   6. 旋转变换/反射变换 Rotation/reflection
   author: Yawei Li
   date:2017-04-15
"""

import os
import math
import random
import glob
import numpy as np
from scipy import misc
from PIL import Image
import cv2
 
# 左右翻转
def flip_left_right(image):
    return image[:, -1::-1]
 
# 改变亮度
def random_brightness(image, max_delta=63, seed=None):
    img = np.array(image)
    delta = np.random.uniform(-max_delta, max_delta)
    image = Image.fromarray(np.uint8(img + delta))
    return image

# 改变旋转
def random_Rotation(image):
    """
     对图像进行随机任意角度(0~360度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    :param image PIL的图像image
    :return: 旋转转之后的图像
    """

    image=cv2.flip(image, 180)
    return image

# 改变对比度
def random_contrast(image, lower, upper, seed=None):
    factor = np.random.uniform(-lower, upper)
    mean = (image[0] + image[1] + image[2]).astype(np.float32) / 3
    img = np.zeros(image.shape, np.float32)
    for i in range(0, 3):
        img[i] = (img[i] - mean) * factor + mean
    return img

# 裁剪图片
def crop(image, name, crop_size, padding_size):
    (width, height) = image.shape
    cropped_images = []
    for i in range(0, width, padding_size):
        for j in range(0, height, padding_size):
            box = (i, j, i+crop_size, j+crop_size) #left, upper, right, lower
            cropped_name = name + '_' + str(i) + '_' + str(j) + '.jpg'
            cropped_image = image[i:i+crop_size, j:j+crop_size]
            resized_image = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
            cropped_images.append(resized_image)
 
    return cropped_images




# 数据扩增
# 将选取的图片文件进行「左右翻转」「改变亮度」「改变对比度」「裁剪」操作data_num次
def data_augmentation(image_files, data_num):
    image_list = []
    file_num = len(image_files)
 
    for image_file in image_files:
        image_list.append(misc.imread(image_file))
 
    if file_num > data_num:
        return image_list

    for image in image_list:
        rotate_image = random_Rotation(image)
        image_list.append(rotate_image)

        if len(image_list) == data_num:
            return image_list


    return image_list
 
 
dir_list = os.listdir("/home/changlu/tmp/u-net_db/TNBC/data")
IMAGE_SIZE =256

# for dir in dir_list:
image_files = glob.glob("/home/changlu/tmp/u-net_db/TNBC/data/*_mask.png")

image_list = data_augmentation(image_files, 2)
print(len(image_list))

for i, image in enumerate(image_list):
    misc.imsave(os.path.join("/home/changlu/tmp/u-net_db/TNBC/Aug", str(i) + '_mask.png'), image)
