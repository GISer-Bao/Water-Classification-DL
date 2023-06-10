# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:40:26 2022

@author: Administrator
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from plot_map import show_mask


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
        show_mask(images[i], images_name[i],fontsize)



from extract_edge import extract_edge

path = r"E:\gaofen-competition\GaoFen_challenge_TrVa\train_validation\Masks\180_img.png"
# label = np.array(Image.open(path))
# show_mask(label,"label")
# lb = laplacian(label) * 255
# show_mask(lb, "edge")

label = np.array(Image.open(path))
grad = extract_edge(label)
laplacian = laplacian(label) * 255

labels = [label, grad, laplacian]
names = ['label', 'gradient', 'laplacian']
plot_maps(images=labels, images_name=names)

ids = os.path.basename(path)[:-4]
plt.savefig(r'C:\Users\DELL\Desktop\picture\1122\{}.png'.format(ids), bbox_inches='tight')



# path = r"E:\GaoFen_challenge\GaoFen_challenge_Te\Masks\84_img.png"
# label = torch.from_numpy(np.array(Image.open(path)))
# label = label.unsqueeze(0).unsqueeze(0)
# print(label.shape)

# radius = 1
# ks = 2 * radius + 1
# filt1 = torch.ones(1, 1, ks, ks)
# filt1[:, :, radius:2*radius, radius:2*radius] = -8

# filt1.requires_grad = False
# lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=radius)
# lbedge = 1 - torch.eq(lbedge, 0).float()
# lbedge = lbedge.numpy().squeeze()
# print(lbedge)




