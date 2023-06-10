
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_map import plot_maps, show_mask


#### 单个图像
# if __name__ == '__main__':
    
#     img_name = "img20_11_2.tif"
#     mask_format = '.tif'
#     mask_name = img_name[:-4] + mask_format
    
#     #### 加载 img 和 mask
#     # import img path
#     img_path = r"E:\GID_experiment\allocate_data_492\test\image_NirRGB" + os.sep + img_name
    
#     # true label
#     true_mask_path = r"E:\GID_experiment\allocate_data_492\test\manual_label"  + os.sep + img_name
#     true_mask = np.array(cv2.imread(true_mask_path, -1))
    
#     # raw mask
#     raw_mask_path = r"E:\GID_experiment\allocate_data_492\test\raw_label"  + os.sep + img_name
#     raw_mask = np.array(cv2.imread(raw_mask_path, -1))
    
#     # revise mask
#     revise_mask_path = r"E:\GID_experiment\allocate_data_492\test\sod_isnet_4_2000_raw_crf"  + os.sep + img_name
#     revise_mask = np.array(cv2.imread(revise_mask_path, -1))
    
#     # eanet
#     eanet_path = r"E:\GID_experiment\allocate_data_492\test\test_result_eanet_131" + os.sep + mask_name
#     eanet = np.array(cv2.imread(eanet_path, -1))
    
#     # u2net
#     u2net_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_reviselabel_240_crf" + os.sep + img_name
#     u2net_crf = np.array(cv2.imread(u2net_path, -1))

#     # u2net + multiple-sides + simple-supervision
#     u2net_ealoss_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_gf_ealoss_simplesupervision_epoch6" + os.sep + img_name
#     u2net_ealoss = np.array(cv2.imread(u2net_ealoss_path, -1))
    
#     # u2net + single-side + simple-supervision
#     u2net_ealossds_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_gf_ealoss_simplesupervision_simpleout_epoch16" + os.sep + img_name
#     u2net_ealossds = np.array(cv2.imread(u2net_ealossds_path, -1))
    
#     img = [img_path, true_mask, raw_mask, revise_mask, eanet, u2net_crf, u2net_ealoss, u2net_ealossds]
#     name = ['image (false color)','true label', 'original label', 'corrected label','EaNet', 'U2Net', 'u2net(MSi+SSu)','u2net(SSi+SSu)']
#     plot_maps(images=img, images_name=name, row_number=1, fontsize=14)

#     plt.savefig(r'C:\Users\DELL\Desktop\picture\1227\{}.png'.format(img_name[:-4]), bbox_inches='tight')

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
        
        # revise mask
        revise_mask_path = r"E:\GID_experiment\allocate_data_492\test\sod_isnet_4_2000_raw_crf"  + os.sep + mask_name
        revise_mask = np.array(cv2.imread(revise_mask_path, -1))
        
        # eanet
        eanet_path = r"E:\GID_experiment\allocate_data_492\test\test_result_eanet_131" + os.sep + mask_name
        eanet = np.array(cv2.imread(eanet_path, -1))
        
        # u2net
        u2net_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_reviselabel_240_crf" + os.sep + img_name
        u2net_crf = np.array(cv2.imread(u2net_path, -1))
    
        # u2net + multiple-sides + simple-supervision
        u2net_ealoss_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_gf_ealoss_simplesupervision_epoch6" + os.sep + img_name
        u2net_ealoss = np.array(cv2.imread(u2net_ealoss_path, -1))
        
        # u2net + single-side + simple-supervision
        u2net_ealossds_path = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_gf_ealoss_simplesupervision_simpleout_epoch16" + os.sep + img_name
        u2net_ealossds = np.array(cv2.imread(u2net_ealossds_path, -1))
        
        img = [img_path, true_mask, raw_mask, revise_mask, eanet, u2net_crf, u2net_ealoss, u2net_ealossds]
        name = ['image (false color)','true label', 'original label', 'corrected label','EaNet', 'U2Net', 'u2net(MSi+SSu)','u2net(SSi+SSu)']
    
        plot_maps(images=img, images_name=name, row_number=1, fontsize=14)
    
        plt.savefig(r'C:\Users\DELL\Desktop\picture\0208\{}.png'.format(img_name[:-4]), bbox_inches='tight')
        