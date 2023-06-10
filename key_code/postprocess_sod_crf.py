
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from plot_map import show_mask, show_img, show_img_tc, show_img_fc, plot_maps
from postprocess_sod_ostu import postprocess_sod_ostu

def crf_processing(image, label, nclasses=2, crf_infer_steps=5, soft_label=True):
    
    image = np.ascontiguousarray(image)
    H, W = image.shape[0], image.shape[1]
    if (len(image.shape) < 3):
        # real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2RGB)
        raise ValueError("The input image should be RGB image.")
    
    if not soft_label:
        unary = unary_from_labels(label, nclasses, gt_prob=0.9, zero_unsure=False)
    else:
        if len(label.shape)==2:
            p_neg = 1.0 - label
            label = np.concatenate((p_neg[...,np.newaxis], label[...,np.newaxis]), axis=2)
        label = label.transpose((2,0,1))
        unary = unary_from_softmax(label)
    unary = np.ascontiguousarray(unary)
    
    crf = dcrf.DenseCRF2D(W, H, nclasses)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=(3,3), compat=3)
    crf.addPairwiseBilateral(sxy=(40, 40), srgb=(5, 5, 5), rgbim=image, compat=10)
    
    # crf.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # crf.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=real_image,
    #                        compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    crf_out = crf.inference(crf_infer_steps)

    # Find out the most probable class for each pixel.
    Q = np.argmax(np.array(crf_out), axis=0).reshape((H, W))
    return Q


if __name__ == '__main__':
    
    ids = '\img8_13_2.tif'
    mask_format = '.png'
    
    # 计算CRF
    img_path = r"E:\GID\train_data_480\filter_contain_water\test\Images_RGB" + ids
    image = cv2.imread(img_path,-1)
    sod_path = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_4_2000\DIS5K-TE1" + ids[:-4] + mask_format
    sod = np.array(cv2.imread(sod_path, -1))
    sod_prob = sod / 255
    label = crf_processing(image=image, label=sod_prob, nclasses=2, crf_infer_steps=5)
    label_crf = np.array(label) * 255
    
    # ostu
    ostu = postprocess_sod_ostu(sod_path) * 255

    # 加载 img 和 mask
    ig_path = r"E:\GID\train_data_480\filter_contain_water\test\Images_NirRGB" + ids
    mask_path = r"E:\GID\train_data_480\filter_contain_water\test\Masks"  + ids
    mask = np.array(cv2.imread(mask_path, -1))
    
    img = [ig_path, ig_path, mask, sod, ostu, label_crf]
    name = ['image (true color)', 'image (false color)','Ground truth', 'SOD', 'OSTU', 'CRF']
    
    plot_maps(images=img, images_name=name, row_number=1, fontsize=14)
    nm = os.path.basename(img[1])[:-4]
    plt.savefig(r'C:\Users\DELL\Desktop\picture\1031\crf\{}.png'.format(nm), bbox_inches='tight')
    
