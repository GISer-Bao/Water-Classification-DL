

import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

def cvt_FileFormat(filesDir: str, outputDir: str, old_format: str='.jpg', new_format: str='.tif'):
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    files = [i for i in os.listdir(filesDir) if i.endswith(old_format)]
    print('\nThe total number of images to be converted format: {}\n'.format(len(files)))
    
    for index,file in tqdm(enumerate(files), total=len(files), desc='Precessing'):
        name = file.split('.')[0]
        old_path = os.path.join(filesDir, file)
        new_path = os.path.join(outputDir, name + new_format)
        img = cv2.imread(old_path, -1)
        cv2.imwrite(new_path, img)
        # img = Image.open(old_path)
        # img.save(new_path, format='TIFF')


if __name__ == '__main__':
    
    filesDir = r"E:\gaofen-competition\GaoFen_challenge\Masks"
    outputDir = r"E:\gaofen-competition\GaoFen_challenge\Masks_tif"
    
    cvt_FileFormat(filesDir, outputDir, old_format='.png')


