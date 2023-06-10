
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

# 计算混淆矩阵
def generate_matrix(num_class,gt_image, pre_image):
    # ground truth中所有正确的值在[0, classe_num])的像素label的mask
    mask = (gt_image >= 0) & (gt_image < num_class)  
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1，这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # (n, n)
    return confusion_matrix
 
def miou(hist):
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return miou

def dice_coeff(pred, target):
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
 
    pre = np.array([[0, 0, 0, 2],
                  [0, 0, 2, 1],
                  [1, 1, 1, 2],
                  [1, 0, 1, 2]])
 
    gt = np.array([[0, 0, 0, 2],
                  [0, 0, 2, 1],
                  [1, 1, 1, 0],
                  [1, 0, 1, 2]])
    #求混淆矩阵
    hist = generate_matrix(3,gt,pre)
    #画混淆矩阵热力图
    ax = plt.axes()
    class_names = ['person', 'dog', 'cat']
    sns.heatmap(hist, annot=True,
                annot_kws={"size": 10},
                xticklabels=class_names,
                yticklabels=class_names, 
                ax=ax)
    ax.set_title('Confusion matrix')
    # plt.savefig('./confusion.jpg')
    plt.show()
    
    #计算miou
    miou_res = miou(hist)
    print(miou_res)



