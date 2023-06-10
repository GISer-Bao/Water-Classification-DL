
import os
import time
import numpy as np
from osgeo import gdal

#  保存tif文件函数
def writeTiff(im_data, file_name, im_geotrans):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(file_name, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


'''
滑动窗口裁剪函数：
inputDir 影像目录
outputDir 裁剪后保存目录
inputFormat 输入图像格式
outputFormat 输出图像格式
xsize 裁剪尺寸（x轴）
ysize 裁剪尺寸（y轴）
xRepetitionRate 重复率(x轴)
yRepetitionRate 重复率(y轴)
keep_extra_edge 是否保留多余边界
'''

def split_tif(inputDir: str, outputDir: str,inputFormat:str='.tif',outputFormat:str='.tif', xsize: int=360, ysize: int=340, 
              xRepetitionRate: float=0.0, yRepetitionRate: float=0.0, keep_extra_edge: bool=False):
    start_time = time.time()
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    tifs = [i for i in os.listdir(inputDir) if i.endswith(inputFormat)]
    print('\nThe total number of images to be split: {}\n'.format(len(tifs)))
    
    for index,img in enumerate(tifs):
        print("Number: {}/{}, being segmenting：{}".format(index+1, len(tifs), img))
        time1 = time.time()
        in_ds = gdal.Open(inputDir + "\\" + img)
        if in_ds == None:
            print("File cannot be opened : ", in_ds)  
        
        width = in_ds.RasterXSize  # 获取数据宽度
        height = in_ds.RasterYSize  # 获取数据高度
        ori_transform = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
        im_data = in_ds.ReadAsArray()  # 获取数据

        col_num = (width - xsize * xRepetitionRate) / (xsize * (1 - xRepetitionRate))  # 宽度可以分成几块
        row_num = (height - ysize * yRepetitionRate) / (ysize * (1 - yRepetitionRate))  # 高度可以分成几块
        if keep_extra_edge:
            if col_num != int(col_num):
                col_num += 1
            if row_num != int(row_num):
                row_num += 1     
        col_num = int(col_num)
        row_num = int(row_num)
        print("row_num:%d   col_num:%d" % (row_num, col_num))
        
        for i in range(row_num):  # 从高度下手！！！ 可以分成几块！
            for j in range(col_num):
                offset_y = int(i * ysize * (1 - yRepetitionRate))
                offset_x = int(j * xsize * (1 - xRepetitionRate))
                b_xsize = min(width - offset_x, xsize)
                b_ysize = min(height - offset_y, ysize)
                
                #裁剪影像矩阵,分为单波段和多波段影像
                if (len(im_data.shape) == 2):
                    out_allband = im_data[offset_y:offset_y+b_ysize, offset_x:offset_x+b_xsize]
                else:
                    out_allband = im_data[:, offset_y:offset_y+b_ysize, offset_x:offset_x+b_xsize]
                # print(out_allband.shape)
     
                #### 创建tif
                file_name = outputDir +"\\"+ "img" + str(index+1) + "_" + str(i+1) + "_" + str(j+1) + outputFormat
     
                # 获取原图的原点坐标信息
                top_left_x = ori_transform[0]  # 左上角x坐标
                w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
                top_left_y = ori_transform[3]  # 左上角y坐标
                n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
                top_left_x = top_left_x + offset_x * w_e_pixel_resolution
                top_left_y = top_left_y + offset_y * n_s_pixel_resolution
                dst_transform = (
                top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
                
                writeTiff(out_allband, file_name, dst_transform)

        time2 = time.time() - time1
        print('TimeUsed: {:.0f}m {:.2f}s'.format(time2//60,time2%60))
    
    end_time = time.time() - start_time
    print('\nTotal Time Used: {:.0f}m {:.2f}s\n'.format(end_time//60,end_time%60))
    print('Finished')

if __name__ == '__main__':
    inputDir = r"E:\GID_experiment\relabel_5classes"
    outputDir = r"E:\GID_experiment\split_data_492\relabel_5classes"
    split_tif(inputDir, outputDir, inputFormat='.tif',outputFormat='.tif',
              xsize=492, ysize=492, xRepetitionRate=0, yRepetitionRate=0, keep_extra_edge=False)
