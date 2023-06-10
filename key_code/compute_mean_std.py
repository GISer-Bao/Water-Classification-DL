

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import time


def cal_mean_std(imgs_dir: str, img_format: str='.jpg'):
    
    start_time = time.time()
    assert os.path.exists(imgs_dir), f"images dir: '{imgs_dir}' does not exist."

    imgs_name_list = [i for i in os.listdir(imgs_dir) if i.endswith(img_format)]
    
    img_channels = np.array(Image.open(os.path.join(imgs_dir, imgs_name_list[0]))).shape[2]
    print("\nThe number of channels in the image: {} \n".format(img_channels))
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    
    for img_name in tqdm(imgs_name_list, desc='Calculating: '):
        img_path = os.path.join(imgs_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.

        cumulative_mean += np.mean(img,axis=(0,1))
        cumulative_std += np.std(img,axis=(0,1))

    mean = cumulative_mean / len(imgs_name_list)
    std = cumulative_std / len(imgs_name_list)
    print(f"\nmean: {mean}")
    print(f"std: {std}")
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Fininshed')

if __name__ == '__main__':
    
    imgs_dir = r"E:\gaofen-competition\GaoFen_challenge\tif_224\Images"
    cal_mean_std(imgs_dir=imgs_dir, img_format='.tif')



#### GID+GF
# mean: [0.35759439 0.40179919 0.40074271]
# std: [0.12684615 0.11657909 0.10883803]

# mean: [0.357, 0.402, 0.401]
# std: [0.127, 0.117, 0.109]


#### GID
# mean: [0.49645429 0.37036301 0.39000201 0.36229235]
# std: [0.24119444 0.22965162 0.22208599 0.21875433]

# mean: [0.496, 0.370, 0.390, 0.362]
# std: [0.241, 0.229, 0.222, 0.219]


# mean: [0.37035905 0.38999641 0.36228811]
# std: [0.22965559 0.22209171 0.21875883]

# mean: [0.370, 0.390, 0.362]
# std: [0.229, 0.222, 0.219]