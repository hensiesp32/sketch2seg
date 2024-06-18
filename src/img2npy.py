import numpy as np
from PIL import Image
import os

def load_img(img_dir_path,save_path):
    for _,_,files in os.walk(img_dir_path):
        for file in files:
            img_path = f"{img_dir_path}/{file}"
            im = Image.open(img_path)
            im_array = np.array(im)
            im_array = np.squeeze(im_array)
            file_name = file.split(".")[0]
            np.save(os.path.join(save_path,f"{file_name}"+".npy"),im_array)

if __name__  =="__main__":
    image_dir = "/root/code_dir/ControlNet_Seg/datasets/25_test_data/anno"
    save_dir = "/root/code_dir/ControlNet_Seg/datasets/25_test_data/image"
    load_img(image_dir,save_dir)
"""
input:图片路径
output：.npy文件

step 1: 读取图片

step 2: 给定类别

step 3: 保存文件
"""