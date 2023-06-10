
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
        
        # raw mask
        raw_mask_path = r"E:\GID_experiment\allocate_data_492\test\raw_label"  + os.sep + mask_name
        raw_mask = np.array(cv2.imread(raw_mask_path, -1))
        
        # u2net (raw label)
        u2net_raw_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_rawlabel_181_crf" + os.sep + mask_name
        u2net_raw = np.array(cv2.imread(u2net_raw_path, -1))
        
        # fcn
        fcn_path = r"E:\GID_experiment\allocate_data_492\test\test_result_fcn-resnet101_epoch155" + os.sep + mask_name
        fcn = np.array(cv2.imread(fcn_path, -1))
        
        # deeplab
        deeplab_path = r"E:\GID_experiment\allocate_data_492\test\test_result_deeplab-resnet101_epoch137"  + os.sep + mask_name
        deeplab = np.array(cv2.imread(deeplab_path, -1))
        
        # unet
        unet_path = r"E:\GID_experiment\allocate_data_492\test\test_result_unet_epoch130" + os.sep + mask_name
        unet = np.array(cv2.imread(unet_path, -1))
        
        # u2net (revise label)
        u2net_revise_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_reviselabel_16_crf"  + os.sep + mask_name
        u2net_revise = np.array(cv2.imread(u2net_revise_path, -1))
        
        # eanet
        eanet_path = r"E:\GID_experiment\allocate_data_492\test\test_result_eanet_131"  + os.sep + mask_name
        eanet = np.array(cv2.imread(eanet_path, -1))
        
        # BDnet
        senet_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_gf_ealoss_simplesupervision_simpleout_epoch16"  + os.sep + mask_name
        senet = np.array(cv2.imread(senet_path, -1))
        
        img = [img_path, true_mask, raw_mask, u2net_raw, fcn, deeplab, unet, u2net_revise, eanet, senet]
        name = ['原始影像','真实标签', '初始标签', '$U^2-Net$\n(初始标签)', 'FCN','DeepLabV3+', 'U-Net', '$U^2-Net$', 'Ea-Net', '本文方法']
    
        plot_maps(images=img, images_name=name, row_number=1, fontsize=14)
    
        plt.savefig(r'D:\硕士论文\gaofen_picture_0309\comparation\{}.png'.format(img_name[:-4]), bbox_inches='tight')
        