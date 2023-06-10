
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from skimage import io
from tqdm import tqdm

############### evaluate metric ###############
## precision, recall, F1-score, overall accuracy , mIOU
# def generate_matrix(num_class,gt_image, pre_image):
    
#     gt_image = normalize(io.imread(gt_image).reshape(-1))
#     pre_image = normalize(io.imread(pre_image).reshape(-1))
#     # ground truth中所有正确的值在[0, classe_num])的像素label的mask
#     mask = (gt_image >= 0) & (gt_image < num_class)  
#     label = num_class * gt_image[mask].astype('int') + pre_image[mask]
#     # np.bincount计算了从0到n**2-1，这n**2个数中每个数出现的次数，返回值形状(n, n)
#     count = np.bincount(label, minlength=num_class ** 2)
#     confusion_matrix = count.reshape(num_class, num_class)  # (n, n)
#     return confusion_matrix

# def precision_recall_miou_f1(gt, mask):
#     mask = normalize(io.imread(mask).reshape(-1))
#     gt = normalize(io.imread(gt).reshape(-1))
#     pre, rec, f1, sup = precision_recall_fscore_support(gt, mask)
#     # precision
#     mp = np.nanmean(pre)
#     # recall
#     mr = np.nanmean(rec)
#     # f1
#     mf1 = np.nanmean(f1)
#     return mp, mr, mf1

# 归一化到[0, 1]
def normalize(mask):
# input 'mask': HxW, output: HxW [0,1]
    if np.amax(mask) != 0:
        mask = mask / (np.amax(mask))
    else:
        mask = mask
    return mask.astype(int)

# 获取list的非零元素
def nonzero(list_number):
    return list_number.ravel()[np.flatnonzero(list_number)].tolist()

def generate_matrix(gt_image, mask_image):
# input 'gt_image': ground truth image path
# input 'mask_image': predict image path
    gt = normalize(io.imread(gt_image).reshape(-1))
    mask = normalize(io.imread(mask_image).reshape(-1))
    cm = confusion_matrix(gt, mask)
    if cm.shape[1] == 1:
        if np.unique(gt)==0 and np.unique(mask)==0:
            confusion = np.zeros((2,2))
            confusion[0][0] = cm[0][0]
        elif np.unique(gt)==1 and np.unique(mask)==1:
            confusion = np.zeros((2,2))
            confusion[1][1] = cm[0][0]
        return confusion.astype(int)
    return cm.astype(int)

def generate_matrix_batch(gt_dir, gt_format, mask_dir, mask_format):
# input 'gt_dir': ground truth images folder
# input 'mask_dir': predict images folder
    img_name = os.listdir(gt_dir)
    hists = []
    for name in tqdm(img_name, total = len(img_name), desc='Precessing'):
        gt = os.path.join(gt_dir, name[:-4] + gt_format)
        mask = os.path.join(mask_dir, name[:-4] + mask_format)
        hist = generate_matrix(gt, mask)
        hists.append(hist)
    total_cm = np.sum(np.array(hists), axis=0)
    return total_cm

def plot_confusion_matrix(hist):
# input 'hist': confusion_matrix
# 画混淆矩阵热力图
    ax = plt.axes()
    class_names = ['non-water', 'water']
    sns.heatmap(hist, annot=True,
                annot_kws={"size": 10},
                xticklabels=class_names,
                yticklabels=class_names, 
                ax=ax)
    ax.set_title('Confusion matrix')
    # plt.savefig('./confusion.jpg')
    plt.show()

def precision_recall_miou_f1_oa(hist):
# input 'hist': confusion_matrix
    # precision
    precision = np.diag(hist) / hist.sum(axis=0)
    mp = np.nanmean(precision)
    # recall
    recall = np.diag(hist) / hist.sum(axis=1)
    mr = np.nanmean(recall)
    # miou
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    # f1
    f1 = (2*precision*recall) / (precision + recall)
    mf1 = np.nanmean(f1)
    # Overall Accuracy
    oa = np.sum(np.diag(hist)) / np.sum(hist)
    return mp, mr, miou, mf1, oa


def dice_coeff(target, pred):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
 
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


if __name__ == '__main__':

    # mask = r"E:\gaofen-competition\GaoFen_challenge_Te\test_result_u2net_ealoss_epoch6_crf\480_img.png"
    # gt = r"E:\gaofen-competition\GaoFen_challenge_Te\Masks\480_img.png"
    
    # 任意格式
    gt_dir = r"E:\GID_experiment\allocate_data_492\test\manual_label"
    mask_dir = r"E:\GID_experiment\allocate_data_492\test\test_result_u2net_rawlabel_181_crf"
    # mask_dir = r"E:\GID_experiment\allocate_data_492\test\raw_label"
    # mask_dir = r"E:\GID_experiment\allocate_data_492\test\sod_isnet_4_2000_raw_crf"


    hist = generate_matrix_batch(gt_dir=gt_dir, gt_format='.tif', mask_dir=mask_dir, mask_format='.tif')
    print("\n",hist)
    
    # 画混淆矩阵热力图
    # plot_confusion_matrix(hist)
    # 计算miou
    
    print('\ndataset: ', mask_dir.split('\\')[-1])
    mp, mr, miou, mf1, oa = precision_recall_miou_f1_oa(hist)
    print("\nprecision:", np.round(mp*100, 3), "\nrecall:", np.round(mr*100, 3), 
          "\nf1-score:", np.round(mf1*100, 3), "\nOverall Accuracy:", np.round(oa*100, 3), 
          "\nmiou:", np.round(miou*100, 3))
    
    





