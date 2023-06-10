
import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import operator
from tqdm import tqdm
import time


def merge_img_mask_png(img_path, save_folder):
    img_name = os.path.basename(img_path)
    # 新
    img = np.array(Image.open(img_path)) * 255
    # merge_img = (img1+img2)*255
    Image.fromarray(img.astype('uint8')).save(os.path.join(save_folder, img_name))

def merge_img_mask_tif(img_path, save_folder):
    img_name = os.path.basename(img_path)
    # 新
    img = np.array(Image.open(img_path))
    # merge_img = (img1+img2)*255
    Image.fromarray(img.astype('uint8')).save(os.path.join(save_folder, img_name[:-4]+'.tif'))

    
if __name__ == '__main__':
    
    start_time = time.time()
    
    root_path = r"E:\GID_experiment\allocate_data_492\test\sod_isnet_4_2000_raw_crf3"
    save_path = r"E:\GID_experiment\allocate_data_492\test\sod_isnet_4_2000_raw_crf3"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_files = os.listdir(root_path)
    print('\nThe total number of images: {}\n'.format(len(image_files)))
    
    for image_file in tqdm(image_files, desc='Processing (train): '):
        img_path = os.path.join(root_path, image_file)
        # merge_img_mask_png(img_path, save_path)        
        merge_img_mask_tif(img_path, save_path)
    
    end_time = time.time() - start_time
    print('\n\nTotal Time Used: {:.0f}m {:.2f}s'.format(end_time//60,end_time%60))
    print('Fininshed')


