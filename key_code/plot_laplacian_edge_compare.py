

import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from plot_map import show_mask, show_img
from extract_edge import extract_edge

# label: np.array, [x_size,y_size]
def laplacian(label, radius=1):
    label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)

    ks = 2 * radius + 1
    filt1 = torch.ones(1, 1, ks, ks)
    filt1[:, :, radius:2*radius, radius:2*radius] = -8

    filt1.requires_grad = False
    lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=radius)
    lbedge = 1 - torch.eq(lbedge, 0).float()
    lbedge = lbedge.numpy().squeeze()
    
    return lbedge

import matplotlib.pyplot as plt
import math
def plot_maps(images:list, images_name:list, row_number:int=1, fontsize=32):
    img_size = len(images)
    plt.figure(figsize=(20, 15))
    for i in range(img_size):
        # plt.subplot(2, img_size, i+1)
        plt.subplot(row_number, math.ceil(img_size/row_number), i+1)
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.2) 
        if len(images[i].shape) == 3:
            show_img(images[i], images_name[i], fontsize)
        else:
            show_mask(images[i], images_name[i],fontsize)



if __name__ == '__main__':
    
    img_name = "847_img.png"
    
    image = r"E:\gaofen-competition\GaoFen_challenge_Te\Images"
    image = os.path.join(image, img_name[:-4] + '.jpg')
    label = r"E:\gaofen-competition\GaoFen_challenge_Te\Masks"
    label = os.path.join(label, img_name)
    ealoss_label = r"E:\gaofen-competition\GaoFen_challenge_Te\test_result_unet_ealoss"
    ealoss_label = os.path.join(ealoss_label, img_name)
    no_ealoss_label = r"E:\gaofen-competition\GaoFen_challenge_Te\test_result_unet_1124"
    no_ealoss_label = os.path.join(no_ealoss_label, img_name)

    image = np.array(Image.open(image))
    label = np.array(Image.open(label))
    label_edge = laplacian(label) * 255
    ealoss_label = np.array(Image.open(ealoss_label))
    ealoss_label_eage = laplacian(ealoss_label) * 255
    no_ealoss_label = np.array(Image.open(no_ealoss_label))
    no_ealoss_label_edge = laplacian(no_ealoss_label) * 255

    labels = [image, label, label_edge, no_ealoss_label, no_ealoss_label_edge, ealoss_label, ealoss_label_eage]
    names = ['image', 'true label', 'true label edge', 'label (w/o EAloss)', 'label edge (w/o EAloss)', 
              'label (with EAloss)', 'label edge (with EAloss)']
    
    plot_maps(images=labels, images_name=names, fontsize=12)
    
    ids = img_name[:-4]
    plt.savefig(r'C:\Users\DELL\Desktop\picture\1124\{}.png'.format(ids), bbox_inches='tight')







