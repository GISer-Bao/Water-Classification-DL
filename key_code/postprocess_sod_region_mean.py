
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from plot_map import show_mask, show_img, show_img_tc, show_img_fc, plot_maps

def postprocess_sod_RegionMean(sod_path: str, mask_path: str, weight: float=1.0):
    
    mask = np.array(cv2.imread(mask_path, -1))
    sod = np.array(cv2.imread(sod_path, -1))
    m = np.mean(sod[mask==255])*weight
    final = np.array((sod>m).astype(np.uint8)) * 255
    
    return final


if __name__ == '__main__':
    
    ids = '\img3_17_3.tif'
    
    image_path = r"E:\GID\train_data_480\filter_contain_water\test\Images_NirRGB" + ids
    mask_path = r"E:\GID\train_data_480\filter_contain_water\test\Masks" + ids
    # sod_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_4_6000\DIS5K-TE1" + name[:-4] + '.png'
    sod_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_4_6000\DIS5K-TE1" + ids[:-4] + '.png'
    
    
    mask = np.array(cv2.imread(mask_path, -1))
    sod = np.array(cv2.imread(sod_path, -1))
    post_sod = postprocess_sod_RegionMean(sod_path, mask_path, weight=1.0)
    post_sod5 = postprocess_sod_RegionMean(sod_path, mask_path, weight=0.5)
    post_sod75 = postprocess_sod_RegionMean(sod_path, mask_path, weight=0.75)
    
    # img = [image_path, image_path, mask, sod, post_sod, post_sod5, post_sod75]
    # name = ['image (true color)', 'image (false color)','Mask', 'SOD', 'Post-SOD(mean)', 'Post-SOD(mean*0.5)','Post-SOD(mean*0.75)']
    
    img = [image_path, image_path, mask, sod, post_sod, post_sod5]
    name = ['image (true color)', 'image (false color)','Mask', 'SOD', 'Post-SOD(mean)', 'Post-SOD(mean*0.5)']
    
    plot_maps(images=img, images_name=name, row_number=1, fontsize=14)
    
    # nm = os.path.basename(img[1])[:-4]
    # plt.savefig(r'C:\Users\DELL\Desktop\picture\0929\{}.png'.format(nm), bbox_inches='tight')