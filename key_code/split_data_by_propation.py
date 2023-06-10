
import os
import time
import random
import shutil
from tqdm import tqdm

'''
划分数据为训练集和测试集：
imgs_path 影像所在目录
masks_path 标签所在目录
img_format 图像格式
mask_format 标签格式
rate 测试集的占比，训练集的占比 "1-rate"
'''

def split_train_test(imgs_path: str, masks_path: str, img_format: str, mask_format: str, rate: float=0.2):
    
    start_time = time.time()
    
    # 生成存放数据的目录
    root = os.path.dirname(os.path.dirname(imgs_path))
    train_path = os.path.join(root,'train_val')
    test_path = os.path.join(root,'test')
    train_images_path = os.path.join(train_path,'Images')
    if not os.path.exists(train_images_path):
        os.makedirs(train_images_path)
        print('create directory: ', train_images_path)
    train_masks_path = os.path.join(train_path,'Masks')
    if not os.path.exists(train_masks_path):
        os.makedirs(train_masks_path)
        print('create directory: ', train_masks_path)
    test_images_path = os.path.join(test_path,'Images')
    if not os.path.exists(test_images_path):
        os.makedirs(test_images_path)
        print('create directory: ', test_images_path)
    test_masks_path = os.path.join(test_path,'Masks')
    if not os.path.exists(test_masks_path):
        os.makedirs(test_masks_path)
        print('create directory: ', test_masks_path)
    
    filenames = [file for file in os.listdir(imgs_path) if file.endswith(img_format)]
    print('\nThe total number of images: {}\n'.format(len(filenames)))
    
    random.seed(43)
    random.shuffle(filenames)
    size = int(rate * len(filenames))
    test_filenames = filenames[:size]
    train_filenames = filenames[size:]
    

    for file in tqdm(train_filenames, desc='Processing (train): '):
        input_img = os.path.join(imgs_path, file)
        output_img = os.path.join(train_images_path, file)
        input_mask = os.path.join(masks_path, os.path.splitext(file)[0] + mask_format)
        output_mask = os.path.join(train_masks_path, os.path.splitext(file)[0] + mask_format)
        # shutil.move(input_img,output_img)
        shutil.copy(input_img, output_img)
        shutil.copy(input_mask, output_mask)

    print("{} images were moved to '{}'".format(len(train_filenames),train_images_path))
    print("{} masks were moved to '{}'\n".format(len(train_filenames),train_masks_path))
    
    for file in tqdm(test_filenames, desc='Processing (test): '):
        input_img = os.path.join(imgs_path, file)
        output_img = os.path.join(test_images_path, file)
        input_mask = os.path.join(masks_path, os.path.splitext(file)[0] + mask_format)
        output_mask = os.path.join(test_masks_path, os.path.splitext(file)[0] + mask_format)
        # shutil.move(input_img,output_img)
        shutil.copy(input_img,output_img)
        shutil.copy(input_mask, output_mask)
    
    print("{} images were moved to '{}'".format(len(test_filenames),test_images_path))
    print("{} masks were moved to '{}'".format(len(test_filenames),test_masks_path))
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Fininshed')


if __name__ == '__main__':
    imgs_path = r"E:\GID_experiment\segment_data_492\image_NirRGB"
    masks_path = r"E:\GID_experiment\segment_data_492\relabel_5classes"
    img_format = '.tif'
    mask_format = '.tif'
    split_train_test(imgs_path=imgs_path, masks_path=masks_path, img_format=img_format, mask_format=mask_format, rate=0.2)