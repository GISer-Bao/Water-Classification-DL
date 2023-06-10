
#### 2023.02.24

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_map import plot_maps, show_mask
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

#### 多个图像
if __name__ == '__main__':
    
    # dir_path = r"C:\Users\DELL\Desktop\picture\1227"
    # name = os.listdir(dir_path)
    
    name = ["img5_6_12.png", "img6_6_10.png", "img6_8_3.png", "img6_8_4.png", "img6_11_9.png", "img7_1_5.png", "img14_6_10.png",
            "img15_4_7.png", "img22_4_12.png", "img22_12_4.png", "img5_4_8.png", "img7_5_6.png", "img16_2_10.png", "img22_13_10.png",
            "img22_13_6.png"]

    for img_name in name:
        img_format = '.tif'
        img_name = img_name[:-4] + img_format
        mask_format = '.tif'
        mask_name = img_name[:-4] + mask_format
        
        img_path = r"E:\GID_experiment\allocate_data_492\test\image_NirRGB" + os.sep + img_name
        
        # true label
        true_mask_path = r"E:\GID_experiment\allocate_data_492\test\manual_label"  + os.sep + mask_name
        true_mask = np.array(cv2.imread(true_mask_path, -1))

        
        img = [img_path, true_mask]
        name = ['原始影像','真实标签']

        plot_maps(images=img, images_name=name, row_number=1, fontsize=40)
    
        plt.savefig(r'D:\硕士论文\gaofen_picture_0309\0519\{}.png'.format(img_name[:-4]), bbox_inches='tight')
        
