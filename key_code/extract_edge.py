# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:26:35 2022

@author: DELL
"""

import cv2
import numpy as np
import os
import time

# gt = (gt > 128);
# gt = double(gt);

def extract_edge(mask: str):

    gt = (np.array(mask) > 128).astype(np.double)
    gy, gx = np.gradient(gt)
    temp_edge = gy*gy + gx*gx
    temp_edge[temp_edge!=0]=1;
    bound = (temp_edge*255).astype(np.uint8)
    return bound

def extract_edge_png(mask_path: str):
    
    edge_dir = os.path.dirname(os.path.dirname(mask_path)) + "\\Masks_edge"
    if not os.path.exists(edge_dir):
        os.makedirs(edge_dir)
    mask_name = mask_path.split("\\")[-1].split(".")[0]
    
    gt = (np.array(cv2.imread(mask_path, -1)) > 128).astype(np.double)
    gy, gx = np.gradient(gt)
    temp_edge = gy*gy + gx*gx
    temp_edge[temp_edge!=0]=1;
    bound = (temp_edge*255).astype(np.uint8)
    cv2.imwrite(os.path.join(edge_dir, mask_name+'.png'), bound)
    
    
if __name__ == '__main__':
    
    import glob
    from tqdm import tqdm
    
    print('\n-------------Processing Start!!!-------------\n')
    start_time = time.time()
    maskDir = r"E:\GID\train_data_480\filter_contain_water\train_val\Masks"
    mask_name_list = glob.glob(maskDir + '\\*.tif')
    for mask_path in tqdm(mask_name_list, total = len(mask_name_list), desc='Precessing'):
        extract_edge_png(mask_path)
    
    print('\n\n-------------Congratulations! Processing Done!!!-------------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))
    


# mask = r"E:\GID\train_data_480\filter_contain_water\test\Masks\img1_5_18.tif"
# gt = (np.array(cv2.imread(mask, -1)) > 128).astype(np.double)

# gy, gx = np.gradient(gt)
# temp_edge = gy*gy + gx*gx
# temp_edge[temp_edge!=0]=1;
# bound = (temp_edge*255).astype(np.uint8)
# print(bound.shape)
# print(np.unique(bound))
# cv2.imshow('edge', bound)
# k = cv2.waitKey(0) 
# if k ==27:
#     cv2.destroyAllWindows() 



