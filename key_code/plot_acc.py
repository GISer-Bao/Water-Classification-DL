import os
import matplotlib.pyplot as plt

def get_acc(path: str,ordinal: int = 9):
    # 读取文件
    lines = open(path,'r').readlines()
    # 删除 "\n"
    info = [line.strip('\n') for line in lines if line.strip('\n') != '']
    acc = []
    for i in range(ordinal,len(info),10):
        num = info[i].split(':')[1].strip()
        acc.append(float(num))
    return acc

def save_plots(fcn_acc, unet_acc, deeplab_acc, fcn101_acc, deeplab101_acc, acc_name: str):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # loss plots
    plt.figure(figsize=(10, 7))
    
    plt.plot(unet_acc, color='red', linestyle='-', marker='v', label='Unet')
    plt.plot(fcn_acc, color='blue', linestyle=':', marker='o', label='fcn_resnet50')
    plt.plot(fcn101_acc, color='cyan', linestyle=':', marker='s', label='fcn_resnet101')
    plt.plot(deeplab_acc, color='green', linestyle='-.', marker='*', label='deeplab-v3_resnet50')
    plt.plot(deeplab101_acc, color='m', linestyle='-.', marker='x', label='deeplab-v3_resnet101')
    
    plt.tick_params(labelsize=18)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(acc_name, fontsize=20)
    plt.title(acc_name, fontsize=26)
    plt.legend(prop={'size':20})
    # plt.savefig(os.path.join(data_dir, acc_name+".png"))


if __name__ == '__main__':
    
    data_dir = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val"
    
    fcn_resnet50_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet50_20220614-180240.txt"
    unet_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\unet_20220614-204414.txt"
    deeplab_resnet50_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet50_20220615-115011.txt"
    fcn_resnet101_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet101_20220615-091725.txt"
    deeplab_resnet101_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet101_20220614-220834.txt"
    
    position = 8
    fcn_resnet50 = get_acc(path = fcn_resnet50_path, ordinal = position)
    unet = get_acc(path = unet_path, ordinal = position)
    deeplab_resnet50 = get_acc(path = deeplab_resnet50_path, ordinal = position)
    fcn_resnet101 = get_acc(path = fcn_resnet101_path, ordinal = position)
    deeplab_resnet101 = get_acc(path = deeplab_resnet101_path, ordinal = position)
    
    save_plots(fcn_acc = fcn_resnet50, unet_acc = unet, deeplab_acc = deeplab_resnet50,
           fcn101_acc = fcn_resnet101, deeplab101_acc = deeplab_resnet101, acc_name = 'mean IOU')






