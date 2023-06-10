import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from PIL import Image
from src import u2net_full

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(imagePath: str, model_weight: str):
    weights_path = model_weight
    img_path = imagePath
    threshold = 0.5

    prediction_dir = os.path.dirname(os.path.dirname(img_path)) + "\\test_result_u2net_gf_ealoss_simplesupervision_simpleout_rawlabel_epoch240"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    assert os.path.exists(weights_path), f"weights file {weights_path} dose not exists."
    assert os.path.exists(img_path), f"image file {img_path} dose not exists."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # from pil image to tensor and normalize
    origin_img = np.array(Image.open(img_path))
    h, w, b = origin_img.shape
    if b==3:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(480),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(480),
            transforms.Normalize(mean=(0.496, 0.370, 0.390, 0.362),
                                  std=(0.241, 0.229, 0.222, 0.219))])
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, dim=0).to(device)


    # load weights
    model = u2net_full(in_ch=b, out_ch=2)
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, b, img_height, img_width), device=device)
        model(init_img)
        pred = model(img)
        # print(pred.shape)
        pred = pred.argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)
        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 1, 0)
        img_name = img_path.split("\\")[-1].split(".")[0]
        
        # # 保存png
        # cv2.imwrite(os.path.join(prediction_dir, img_name+'.png'), pred*255)
        # 保存tif
        pred = pred*255
        Image.fromarray(pred.astype('uint8')).save(os.path.join(prediction_dir, img_name+'.tif'))



if __name__ == '__main__':
    
    import glob
    import tqdm
    
    print('\n-------------Processing Start!!!-------------\n')
    start_time = time.time()

    imageDir = r"E:\GID_experiment\allocate_data_492\test\image_NirRGB"
    model_weight = r"E:\GID_experiment\allocate_data_492\train_u2net_ealoss\save_weights_u2net_ealoss_simplesupervision_simpleout_rawlabel\u2net_last_model_epoch240_ealoss0.9296542934328318_mae0.7793323733642697_f10.0.pth"
    img_name_list = glob.glob(imageDir + '\\*.tif')
    
    for img_path in tqdm.tqdm(img_name_list, total = len(img_name_list), desc='Precessing'):
        main(imagePath=img_path, model_weight=model_weight)

    print('\n\n-------------Congratulations! Processing Done!!!-------------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))


# if __name__ == '__main__':
#     img_path = "E:\gaofen-competition\GaoFen_challenge_Te\Images\9_img.jpg"
#     model_weight = r"E:\gaofen-competition\GaoFen_challenge_TrVa\save_weights_u2net_ealoss\u2net_best_model_epoch6_ealoss0.9720123711228371_mae0.5501276909660254_f10.3553131846237188.pth"
#     main(imagePath=img_path, model_weight=model_weight)
    
    
