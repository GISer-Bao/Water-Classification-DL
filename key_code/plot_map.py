
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

'''
绘制掩膜（mask）：
mask: 需要绘制的掩膜，格式为数组
mask_name：图名，格式为字符串
fontsize：图名大小，格式为数字
'''
def show_mask(mask:list, mask_name,fontsize=32):
    temp = np.expand_dims(np.array(mask),axis=2).repeat(3,axis=2)
    plt.imshow(temp.astype(np.uint8))
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.title(mask_name,fontsize=fontsize)

'''
绘制图像（image）：
mask: 需要绘制的图像（RGB格式），格式为图像路径（字符串）
mask_name：图名，格式为字符串
fontsize：图名大小，格式为数字
'''
def show_img(img, img_name,fontsize=32):
    temp = np.array(img)
    plt.imshow(temp.astype(np.uint8))
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name,fontsize=fontsize)

'''
绘制图像,真彩色（image）：
mask: 需要绘制的图像（RGBNir格式），格式为图像路径（字符串）
mask_name：图名，格式为字符串
fontsize：图名大小，格式为数字
'''
def show_img_tc(img_path, img_name, fontsize=28):
    temp = np.array(Image.open(img_path))[:,:,1:]
    plt.imshow(temp.astype(np.uint8))
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name,fontsize=fontsize)


'''
绘制图像,假彩色（image）：
mask: 需要绘制的图像（RGBNir格式），格式为图像路径（字符串）
mask_name：图名，格式为字符串
fontsize：图名大小，格式为数字
'''
def show_img_fc(img_path, img_name, fontsize=28):
    temp = np.array(Image.open(img_path))[:,:,:3]
    plt.imshow(temp.astype(np.uint8))
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name,fontsize=fontsize)
    
'''
绘制所有图像,（image，mask）：
images: 图像列表
images_names: 图名列表
row_number: 图像的行数
fontsize：图名大小，格式为数字
'''
# def plot_maps(images:list, images_name:list, row_number:int=1, fontsize=32, save_fig=False):
    
#     img_size = len(images)
#     plt.figure(figsize=(20, 15))
#     for i in range(img_size):
#         # plt.subplot(2, img_size, i+1)
#         plt.subplot(row_number, math.ceil(img_size/row_number), i+1)
#         plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)     
#         if i == 0:
#             show_img_tc(images[i], images_name[i], fontsize)
#         elif i ==1:
#             show_img_fc(images[i], images_name[i],fontsize)
#         else:
#             show_mask(images[i], images_name[i],fontsize)
#     if save_fig:
#         ids = os.path.basename(images[1])[:-4]
#         plt.savefig(r'C:\Users\DELL\Desktop\picture\{}.png'.format(ids), bbox_inches='tight')
        
def plot_maps(images:list, images_name:list, row_number:int=1, fontsize=32, save_fig=False):
    
    img_size = len(images)
    plt.figure(figsize=(20, 15))
    for i in range(img_size):
        # plt.subplot(2, img_size, i+1)
        plt.subplot(row_number, math.ceil(img_size/row_number), i+1)
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)     
        if i == 0:
            show_img_fc(images[i], images_name[i],fontsize)
        else:
            show_mask(images[i], images_name[i],fontsize)
    if save_fig:
        ids = os.path.basename(images[1])[:-4]
        plt.savefig(r'C:\Users\DELL\Desktop\picture\{}.png'.format(ids), bbox_inches='tight')


if __name__ == '__main__':

    ids = '\img1_2_10.tif'
    img_format = '.tif'
    mask_format = '.png'
    list_path = []
    
    img_path1 = r"E:\GID\train_data_480\filter_contain_water\test\Images_NirRGB" + ids
    list_path.append(img_path1)
    
    img_path2 = r"E:\GID\train_data_480\filter_contain_water\test\Images_NirRGB" + ids
    list_path.append(img_path2)

    mask_path = r"E:\GID\train_data_480\filter_contain_water\test\Masks" + ids
    list_path.append(mask_path)
    
    result1_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_u2net_3" + ids[:-4] + mask_format
    list_path.append(result1_path)
    
    result2_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_u2net_4" + ids[:-4] + mask_format
    list_path.append(result2_path)
    
    result3_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_3\DIS5K-TE1" + ids[:-4] + mask_format
    list_path.append(result3_path)
    
    result4_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_4\DIS5K-TE1" + ids[:-4] + mask_format
    list_path.append(result4_path)
    


    title = ['image (true color)', 'image (false color)','Ground truth', 'U2Net (RGB)','U2Net (NirRGB)','ISNet (RGB)', 'ISNet (NirRGB)']
    # title = ['Raw image','Ground truth','Post-processing','U-net']
    plot_maps(images=list_path, images_name=title, row_number=1, fontsize=14)
    # ids = os.path.basename(list_path[1])[:-4]
    # plt.savefig(r'C:\Users\DELL\Desktop\picture\0909\{}.png'.format(ids), bbox_inches='tight')


    # plt.figure(figsize=(20, 15))
    # for i in range(img_size):
    #     plt.subplot(1, img_size, i+1)
    #     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)     
    #     if i == 0:
    #         show_img_tc(list_path[i], title[i])
    #     elif i ==1:
    #         show_img_fc(list_path[i], title[i])
    #     else:
    #         show_mask(list_path[i], title[i])
    
    # plt.figure(figsize=(20, 15))
    # for i in range(img_size):
    #     # plt.subplot(2, img_size, i+1)
    #     plt.subplot(2, math.ceil(img_size/2), i+1)
    #     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2)     
    #     if i == 0:
    #         show_img_tc(list_path[i], title[i])
    #     elif i ==1:
    #         show_img_fc(list_path[i], title[i])
    #     else:
    #         show_mask(list_path[i], title[i])
    # 

