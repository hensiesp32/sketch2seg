import os
import cv2
import numpy as np

# img_dir = "/root/code_dir/ControlNet_retrival/datasets/zero_25_retrival_data/sketch/cabin"
# save_dir = "/root/code_dir/ControlNet_retrival/datasets/256zero_25_retrival_data/sketch/cabin"
# for _,_,files in os.walk(img_dir):
#     for file in files:
#         img_path = f"{img_dir}/{file}"
#         img = cv2.imread(img_path)
#         res = cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
#         save_path = f"{save_dir}/{file}"
#         cv2.imwrite(save_path,res)

def resize_image(old_path,new_path):
    for _,dirs,_ in os.walk(old_path):
        for dir in dirs:
            new_dir_path = os.path.join(new_path,dir)
            if not os.path.exists(new_dir_path):
                os.mkdir(new_dir_path)
            img_path = os.path.join(old_path,dir)
            for _,_,imgs in os.walk(img_path):
                for img in imgs:
                    file_path = f"{img_path}/{img}"
                    file = cv2.imread(file_path)
                    res = cv2.resize(file,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
                    save_path = f"{new_dir_path}/{img}"
                    # import pdb
                    # pdb.set_trace()
                    cv2.imwrite(save_path,res)
    return 0
                    
if __name__ =="__main__":
    photo_path = "/root/code_dir/ControlNet_retrival/datasets/seg_data/test/photo"
    resize_path = "/root/code_dir/ControlNet_retrival/datasets/256_seg_data/test/photo"
    result = resize_image(photo_path,resize_path)


"""
step 1 : 遍历photo文件夹，创建新的photo_class文件夹和sketch_class文件夹

step 2 : 遍历类别文件夹中的图片，resize后写入新的文件夹
"""