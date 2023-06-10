
import time
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def tifmask_to_pngmask(image_path,save_path,scale=1.0):

    img = Image.fromarray(np.array(Image.open(image_path)) * scale)
    # print(img)
    # print(img.dtype)
    filename = os.path.basename(image_path).split('.')[0]
    # print(filename)
    save_path = save_path + '\\' + filename + '.png'
    img.convert("L").save(save_path)

def pngmask_to_tifmask(image_path,save_path,scale=1.0):

    img = Image.fromarray(np.array(Image.open(image_path)) * scale)
    # print(img)
    # print(img.dtype)
    filename = os.path.basename(image_path).split('.')[0]
    # print(filename)
    save_path = save_path + '\\' + filename + '.tif'
    img.convert("L").save(save_path)


# show_format 有两个选择：NirRGB, NirRGB
def tifimg_to_jpgimg(image_path, save_path, show_format='NirRGB'):
    if show_format == 'RGB':
        img = Image.fromarray(np.array(Image.open(image_path))[:,:,1:])
    elif show_format == 'NirRGB':
        img = Image.fromarray(np.array(Image.open(image_path))[:,:,:3])
    filename = os.path.basename(image_path).split('.')[0]
    save_path = save_path + '\\' + filename + '.jpg'
    img.convert("RGB").save(save_path)


if __name__ == '__main__':
    
    # img = r"E:\GID\train_data_480\filter_contain_water\train\Images_NirRGB\label\img10_10_7_mask.tif"
    # save = r"E:\GID\train_data_480\filter_contain_water\train\Images_NirRGB\label"
    # tif_to_png(img, save, 255)
    
    # img = r"E:\gaofen-competition\GaoFen_challenge_Te\Masks\9_img.png"
    # save = r"E:\gaofen-competition\GaoFen_challenge_Te\Masks"
    # png_to_tif(img, save)
    
    start_time = time.time()
    
    root_path = r"E:\GID_experiment\allocate_data_492\test\raw_label"
    save_path = r"E:\GID_experiment\allocate_data_492\test\raw_label_png"
    
    image_files = os.listdir(root_path)
    print('\nThe total number of images: {}\n'.format(len(image_files)))
    
    for image_file in tqdm(image_files, desc='Processing (train): '):
        img_path = os.path.join(root_path, image_file)
        # tifimg_to_jpgimg(img_path, save_path, show_format='RGB')
        tifmask_to_pngmask(img_path, save_path, scale=1.0)
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Fininshed')

