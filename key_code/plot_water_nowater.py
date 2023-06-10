

import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
绘制在影像上水体和非水体的范围 ：
img_path  影像路径
mask_path  掩膜路径

img_mask  水体或非水体范围
'''

def water_extent(img_path: str, mask_path: str):
    img = cv2.imread(img_path,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.array(cv2.imread(mask_path, -1)) / 255
    img_mask = img * np.expand_dims(mask, 2).astype('uint8')
    return img_mask


def nowater_extent(img_path: str, mask_path: str):
    img = cv2.imread(img_path,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = 1 - (np.array(cv2.imread(mask_path, -1)) / 255)
    img_mask = img * np.expand_dims(mask, 2).astype('uint8')
    return img_mask

def image_mask(img_path: str, mask_path: str):
    img = cv2.imread(img_path,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, -1)
    return img, mask


# img_path = r"E:\GID\exp_rgb_s\seg_224\images\img1_1_27.tif"
# mask_path = r"E:\GID\exp_rgb_s\seg_224\masks\img1_1_27.tif"

# water = water_extent(img_path, mask_path)
# nowater = nowater_extent(img_path, mask_path)
# image, mask = image_mask(img_path, mask_path)
# img_list = [image, mask, water, nowater]

# title = ['image','label', 'water','no-water']

# img_num = 4
# plt.figure(figsize=(20, 15))
# for i in range(img_num):
#     plt.subplot(1, img_num, i+1)
#     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)
#     if i == 1:
#         plt.imshow(img_list[i],cmap='gray')
#     else:
#         plt.imshow(img_list[i])
#     plt.axis('on')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title[i],fontsize=32)
    
img_path = r"E:\GID\exp_rgb_s\small_image_3\img3_4_17.tif"
mask_path = r"E:\GID\exp_rgb_s\small_mask_3\img3_4_17.tif"
salient_path = r"E:\GID\exp_rgb_s\test_reslut_basnet\img3_4_17.tif"
result_path = r"E:\GID\exp_rgb_s\post_mask\img3_4_17.tif"


salient = cv2.imread(salient_path, -1)
result = cv2.imread(result_path, -1)
image, mask = image_mask(img_path, mask_path)
img_list = [image, mask, salient, result]

title = ['image','label', 'salient','exp-result']

img_num = 4
plt.figure(figsize=(20, 15))
for i in range(img_num):
    plt.subplot(1, img_num, i+1)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)
    if i >= 1:
        plt.imshow(img_list[i],cmap='gray')
    else:
        plt.imshow(img_list[i])
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.title(title[i],fontsize=32)






