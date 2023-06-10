
import cv2
import os
import numpy as np
from PIL import Image
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
	#						compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    crf_out = crf.inference(crf_infer_steps)

    # Find out the most probable class for each pixel.
    Q = np.argmax(np.array(crf_out), axis=0).reshape((H, W))
    return Q


if __name__ == '__main__':
    
    import glob
    import time
    from tqdm import tqdm
    
    print('\n-------------Processing Start !!!-------------\n')
    start_time = time.time()

    sodDir = r"E:\GID_experiment\allocate_data_492\label_correction_test\test_result_u2net_240"
    imageDir = r"E:\GID_experiment\allocate_data_492\label_correction_test\image_RGB"
    maskDir = os.path.dirname(imageDir) + os.sep + sodDir.split(os.sep)[-1] + "_crf"     
    if not os.path.exists(maskDir):
        os.makedirs(maskDir)

    name_list = os.listdir(imageDir)
    for name in tqdm(name_list, total = len(name_list), desc='Precessing'):
        image = np.array(Image.open(os.path.join(imageDir,name)))
        sod = np.array(Image.open(os.path.join(sodDir,name[:-4] + '.tif')))
        sod_prob = np.array(sod) / 255
        label = crf_processing(image=image, label=sod_prob, nclasses=2, crf_infer_steps=5)
        label_crf = np.array(label) * 255
        Image.fromarray(label_crf.astype('uint8')).save(os.path.join(maskDir, name[:-4] + '.tif'))

    # name_list = os.listdir(imageDir)
    # for name in tqdm(name_list, total = len(name_list), desc='Precessing'):
    #     image = cv2.imread(os.path.join(imageDir,name), -1)
    #     sod = cv2.imread(os.path.join(sodDir,name[:-4] + '.png'), -1)
    #     sod_prob = np.array(sod) / 255
    #     label = crf_processing(image=image, label=sod_prob, nclasses=2, crf_infer_steps=5)
    #     label_crf = np.array(label) * 255
    #     cv2.imwrite(os.path.join(maskDir, name[:-4] + '.png'), label_crf)
    
    print('\n\n-------------Congratulations! Processing Done !!!-------------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))
    
    
    
