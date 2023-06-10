

import os

# 批量文件重命名
imgs_dir = r"E:\gaofen-competition\experiment\images"
masks_dir = r"E:\gaofen-competition\experiment\maskes"

def rename_file(imgs_dir: str):
    for file in os.listdir(imgs_dir):
        old_name = os.path.join(imgs_dir, file)
        new_name = os.path.join(imgs_dir, os.path.splitext(file)[0]+'_img.jpg')
        os.rename(old_name, new_name)
    print('Finished')
    

if __name__ == '__main__':
    
    imgs_dir = r"E:\gaofen-competition\experiment\images"
    masks_dir = r"E:\gaofen-competition\experiment\maskes"
    
    rename_file(imgs_dir = imgs_dir)