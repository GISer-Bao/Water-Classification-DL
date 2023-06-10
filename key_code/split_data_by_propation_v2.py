# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:29:46 2022

@author: DELL
"""

import os
import time
import random
import shutil
from tqdm import tqdm

'''
划分数据为训练集和测试集：
imgs1_path NirRGB影像所在目录
imgs2_path RGB影像所在目录
masks_path 标签所在目录
img1_format NirRGB图像格式
img2_format RGB图像格式
mask_format 标签格式
rate 测试集的占比，训练集的占比 "1-rate"
'''

def split_train_test(imgs1_path: str, imgs2_path: str, masks_path: str, imgs1_format: str, imgs2_format: str, mask_format: str, rate: float=0.2):
    
    start_time = time.time()
    
    # 生成存放数据的目录
    root = os.path.dirname(os.path.dirname(imgs1_path))
    train_path = os.path.join(root,'train_val')
    test_path = os.path.join(root,'test')
    train_images_path1 = os.path.join(train_path,'Images_NirRGB')
    if not os.path.exists(train_images_path1):
        os.makedirs(train_images_path1)
        print('create directory: ', train_images_path1)
    train_images_path2 = os.path.join(train_path,'Images_RGB')
    if not os.path.exists(train_images_path2):
            os.makedirs(train_images_path2)
            print('create directory: ', train_images_path2)
    train_masks_path = os.path.join(train_path,'Masks')
    if not os.path.exists(train_masks_path):
        os.makedirs(train_masks_path)
        print('create directory: ', train_masks_path)

    test_images_path1 = os.path.join(test_path,'Images_NirRGB')
    if not os.path.exists(test_images_path1):
        os.makedirs(test_images_path1)
        print('create directory: ', test_images_path1)
    test_images_path2 = os.path.join(test_path,'Images_RGB')
    if not os.path.exists(test_images_path2):
            os.makedirs(test_images_path2)
            print('create directory: ', test_images_path2)
    test_masks_path = os.path.join(test_path,'Masks')
    if not os.path.exists(test_masks_path):
        os.makedirs(test_masks_path)
        print('create directory: ', test_masks_path)
    
    filenames = [file for file in os.listdir(imgs1_path) if file.endswith(imgs1_format)]
    print('\nThe total number of images: {}\n'.format(len(filenames)))
    
    random.seed(43)
    random.shuffle(filenames)
    size = int(rate * len(filenames))
    test_filenames = filenames[:size]
    train_filenames = filenames[size:]
    

    for file in tqdm(train_filenames, desc='Processing (train): '):
        input_img1 = os.path.join(imgs1_path, file)
        output_img1 = os.path.join(train_images_path1, file)
        
        input_img2 = os.path.join(imgs2_path, file)
        output_img2 = os.path.join(train_images_path2, file)
        
        input_mask = os.path.join(masks_path, os.path.splitext(file)[0] + mask_format)
        output_mask = os.path.join(train_masks_path, os.path.splitext(file)[0] + mask_format)
        # shutil.move(input_img,output_img)
        shutil.copy(input_img1, output_img1)
        shutil.copy(input_img2, output_img2)
        shutil.copy(input_mask, output_mask)

    print("{} NirRGB images were moved to '{}'".format(len(train_filenames),train_images_path1))
    print("{} RGB images were moved to '{}'".format(len(train_filenames),train_images_path2))
    print("{} masks were moved to '{}'\n".format(len(train_filenames),train_masks_path))
    
    for file in tqdm(test_filenames, desc='Processing (test): '):
        input_img1 = os.path.join(imgs1_path, file)
        output_img1 = os.path.join(test_images_path1, file)
        
        input_img2 = os.path.join(imgs2_path, file)
        output_img2 = os.path.join(test_images_path2, file)
        
        input_mask = os.path.join(masks_path, os.path.splitext(file)[0] + mask_format)
        output_mask = os.path.join(test_masks_path, os.path.splitext(file)[0] + mask_format)
        # shutil.move(input_img,output_img)
        shutil.copy(input_img1,output_img1)
        shutil.copy(input_img2,output_img2)
        shutil.copy(input_mask, output_mask)
    
    print("{} NirRGB images were moved to '{}'".format(len(test_filenames),test_images_path1))
    print("{} RGB images were moved to '{}'".format(len(test_filenames),test_images_path2))
    print("{} masks were moved to '{}'".format(len(test_filenames),test_masks_path))
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Fininshed')


if __name__ == '__main__':
    imgs1_path = r"E:\GID_experiment\segment_data_492\image_NirRGB"
    imgs2_path = r"E:\GID_experiment\segment_data_492\image_RGB"
    masks_path = r"E:\GID_experiment\segment_data_492\relabel_5classes"
    imgs1_format = '.tif'
    imgs2_format = '.tif'
    mask_format = '.tif'
    split_train_test(imgs1_path=imgs1_path, imgs2_path=imgs2_path, masks_path=masks_path, imgs1_format=imgs1_format, imgs2_format=imgs2_format, mask_format=mask_format, rate=0.2)