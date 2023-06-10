import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import time
    

def postprocess_otsu(mask_path: str):

    ## new path for mask
    mask_dir = os.path.dirname(os.path.dirname(mask_path))
    mask_name = os.path.basename(mask_path)
    if not os.path.exists(mask_dir + '\\post_mask'):
        os.makedirs(mask_dir + '\\post_mask')
    new_mask_path = mask_dir + '\\post_mask\\' + mask_name

    model_output = np.array(cv2.imread(mask_path, -1))
    threshold, mask = cv2.threshold(model_output,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (model_output > threshold).astype(np.uint8)
    
    # mask = (model_output > 0.5).astype(np.uint8)
    # contours, _ = cv2.findContours(deepcopy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # Throw out small segments
    # for contour in contours:
    #     segment_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    #     segment_mask = cv2.drawContours(segment_mask, [contour], 0, 255, thickness=cv2.FILLED)
    #     area = (np.sum(segment_mask) / 255.0) / np.prod(segment_mask.shape)
    #     if area < 0.00001:
    #         mask[segment_mask == 255] = 0

    cv2.imwrite(new_mask_path, mask*255)
    # return mask

def postprocess_sod_ostu(mask_path: str):
    model_output = np.array(cv2.imread(mask_path, -1))
    threshold, mask = cv2.threshold(model_output,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (model_output > threshold).astype(np.uint8)
    return mask




if __name__ == '__main__':
    print('\n-------------Processing Start!!!-------------\n')
    
    mask_format = '.png'
    mask_dir = r"E:\GID\train_data_480\filter_contain_water\test\test_result_isnet_4_2000\DIS5K-TE1"
    mask_path = glob.glob(mask_dir + '\\*' + mask_format)
    
    start_time = time.time()
    for mask in tqdm(mask_path, desc='Processing'):
        postprocess_otsu(mask)
        
    print('\n\n-------------Congratulations! Processing Done!!!-------------\n')
    
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))



