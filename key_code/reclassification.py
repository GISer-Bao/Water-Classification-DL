import cv2
import os
import time
import numpy as np


'''
重分类函数：

FilePath 输入文件所在目录
SavePath 输出文件所在目录

'''

def reclassify(FilePath: str, SavePath: str):
    if not os.path.exists(FilePath):
        print("The file path does not exist: ", FilePath)
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    
    img_name = [i for i in os.listdir(FilePath) if i.endswith(".tif")]
    print('The total number of images to be reclassified: {}\n'.format(len(img_name)))
    start_time = time.time()
    for index,img in enumerate(img_name):
        print("Number: {}/{}, being reclassify：{}".format(index+1, len(img_name), img))
        time1 = time.time()
        img_inputpath = os.path.join(FilePath, img)
        img_outputpath = os.path.join(SavePath, img)
        img = cv2.imread(img_inputpath)
        b3 = (img[:,:,0] == 255) * 1.0
        b2 = (img[:,:,1] == 0) * 1.0
        b1 = (img[:,:,2] == 0) * 1.0
        water = np.array(((b1+b2+b3) == 3) * 255.0, dtype=np.uint8)
        cv2.imwrite(img_outputpath, water)
        
        time2 = time.time() - time1
        print('Time Used: {:.0f}m {:.2f}s'.format(time2//60,time2%60))
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Finished')


if __name__ == '__main__':
    FilePath = r"E:\GID\label_5classes"
    SavePath = r"E:\GID\relabel"
    reclassify(FilePath, SavePath)