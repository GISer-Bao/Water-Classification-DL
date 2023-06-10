

import os
import time
import random
from tqdm import tqdm
import cv2
import numpy as np
import datetime
import shutil



def filtering_img(filesDir: str):
    
    print('\n-------------Processing Start!---------------\n')
    start_time = time.time()
    
    baseDir = os.path.dirname(filesDir)
    results_file = baseDir + "\\Image_filename_containing_water_{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    files = [i for i in os.listdir(filesDir) if i.endswith('.tif')]
    print('\nThe total number of images to be processed: {}\n'.format(len(files)))

    for index,file in tqdm(enumerate(files), total=len(files), desc='Precessing'):
        old_path = os.path.join(filesDir, file)
        img = cv2.imread(old_path, -1)
        water = np.sum(np.array(img) / 255)
        if water >= 100:
            with open(results_file, "a") as f:
                file_info = f"{file}\n"
                f.write(file_info)
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('\n\n------Congratulations! Processing Done!------')


def copy_file(txt: str, oldDir: str, newDir: str):
    
    print('\n-------------Processing Start!---------------\n')
    start_time = time.time()
    
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    # 读取文件
    lines = open(txt,'r').readlines()
    # 删除 "\n"
    names = [line.strip('\n') for line in lines if line.strip('\n') != '']
    print('\nThe total number of images to be processed: {}\n'.format(len(names)))
    
    for index, name in tqdm(enumerate(names), total=len(names), desc='Precessing'):
        old_path = os.path.join(oldDir, name)
        new_path = os.path.join(newDir, name)
        shutil.copy(old_path, new_path)
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s'.format(end_time//60,end_time%60))
    print('\n\n------Congratulations! Processing Done!------')

def random_select(txt: str, select_number: int):
    
    print('\n-------------Processing Start!---------------\n')
    
    baseDir = os.path.dirname(txt)
    results_file = baseDir + "\\Image_filename_containing_water_select_part_img_{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    no_results_file = baseDir + "\\Image_filename_containing_water_unselect_part_img_{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # 读取文件
    lines = open(txt,'r').readlines()
    # 删除 "\n"
    names = [line.strip('\n') for line in lines if line.strip('\n') != '']
    
    random.seed(43)
    random.shuffle(names)
    select_names = names[:select_number]
    unselect_names = names[select_number:]
    
    for index,file in tqdm(enumerate(select_names), total=len(select_names), desc='Precessing'):
        with open(results_file, "a") as f:
            file_info = f"{file}\n"
            f.write(file_info)
    for index,file in tqdm(enumerate(unselect_names), total=len(unselect_names), desc='Precessing'):
        with open(no_results_file, "a") as f:
            file_info = f"{file}\n"
            f.write(file_info)
    
    print('\n\n------Congratulations! Processing Done!------')



if __name__ == '__main__':
    
    # filesDir = r"E:\GID\train_data_480\label"
    # filtering_img(filesDir)
    
    txt = r"E:\GID\train_data_480\Image_filename_containing_water_20220904-153724.txt"
    oldDir = r"E:\GID\train_data_480\no_filter\label"
    newDir = r"E:\GID\train_data_480\filter_contain_water\label"
    copy_file(txt, oldDir, newDir)
    
    # txt = r"E:\GID\Image_filename_containing_water_20220706-144929.txt"
    # select_number = 11000
    # random_select(txt, select_number)


