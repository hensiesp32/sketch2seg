import os
from get_feature import FeatureExtractorDDPM, collect_features
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import json
# -----------------------------------
# step1 : get image&sketch path/label path
# input : dir
# output : images_path,sketchs_path, List
def get_path(image_path,sketch):
    images_path = []
    sketchs_path = []
    label_path = []
    for _,_,files in os.walk(image_path):
        files = sorted(files)
        for file in files:
            index = file.split(".")[-1]
            if index == "jpg":
                path = f"{image_path}/{file}"
                images_path.append(path)
            elif index == "npy":
                path = f"{image_path}/{file}"
                label_path.append(path)

    for _,_,files in os.walk(sketch):
        files = sorted(files)
        for file in files:
            index = file.split(".")[-1]
            if index == "png":  # 根据实际调整
                path = f"{sketch}/{file}"
                sketchs_path.append(path)

    return images_path,sketchs_path,label_path

def get_one_category_file(photo_path,sketch_path,anno_path):
    img_list = []
    sketch_list = []
    label_list = []
    for _,_,imgs in os.walk(photo_path):
        imgs = sorted(imgs)
        for i in range(6):
            img_path = os.path.join(photo_path,imgs[i])
            img_list.append(img_path)
    for _,_,sks in os.walk(sketch_path):
        sks = sorted(sks)
        for i in range(6):
            sk_path = os.path.join(sketch_path,sks[i])
            sketch_list.append(sk_path)
    for _,_,files in os.walk(anno_path):
        files = sorted(files)
        for i in range(6):
            label_path = os.path.join(anno_path,files[i])
            label_list.append(label_path)
    # import pdb
    # pdb.set_trace()
    return img_list,sketch_list,label_list

def get_one_category_file_2(photo_path,sketch_path,anno_path):
    img_list = []
    sketch_list = []
    label_list = []
    for _,_,imgs in os.walk(photo_path):
        imgs = sorted(imgs)
        for i in range(75,30,55):
            img_path = os.path.join(photo_path,imgs[i])
            img_list.append(img_path)
    for _,_,sks in os.walk(sketch_path):
        sks = sorted(sks)
        for i in range(75,30,55):
            sk_path = os.path.join(sketch_path,sks[i])
            sketch_list.append(sk_path)
    for _,_,files in os.walk(anno_path):
        files = sorted(files)
        for i in range(75,30,55):
            label_path = os.path.join(anno_path,files[i])
            label_list.append(label_path)
    # import pdb
    # pdb.set_trace()
    return img_list,sketch_list,label_list

def get_file_list(photo_path,sketch_path,anno_path):
    # with open(catgory_path,"r") as fp:
    #     catgory_info = json.load(fp)
    img_list = []
    sketch_list = []
    label_list = []
    for _,dirs,_ in os.walk(photo_path):
        for j,dir in enumerate(dirs):
            img_class_dir = os.path.join(photo_path,dir) # ./photo/airplane
            sk_class_dir = os.path.join(sketch_path,dir) # ./sketch/airplane
            label_class_dir = os.path.join(anno_path,dir)
            for _,_,imgs in os.walk(img_class_dir):
                imgs = sorted(imgs)
                for i in range(5):
                    img_path = os.path.join(img_class_dir,imgs[i])
                    img_list.append(img_path)
            for _,_,files in os.walk(label_class_dir):
                files = sorted(files)
                for i in range(5):
                    label_path = os.path.join(label_class_dir,files[i])
                    label_list.append(label_path)
            for _,_,sks in os.walk(sk_class_dir):
                sks = sorted(sks)
                for i in range(5):
                    sk_path = os.path.join(sk_class_dir,sks[i])
                    sketch_list.append(sk_path)
    # import pdb
    # pdb.set_trace()
    return img_list,sketch_list,label_list

# no label 版本

# def get_path(image_path,sketch):
#     images_path = []
#     sketchs_path = []
#     # label_path = []
#     for _,_,files in os.walk(image_path):
#         files = sorted(files)
#         for file in files:
#             index = file.split(".")[-1]
#             if index == "jpg":
#                 path = f"{image_path}/{file}"
#                 images_path.append(path)
#             # elif index == "npy":
#             #     path = f"{image_path}/{file}"
#             #     label_path.append(path)

#     for _,_,files in os.walk(sketch):
#         files = sorted(files)
#         for file in files:
#             index = file.split(".")[-1]
#             if index == "jpg":  # 根据实际调整
#                 path = f"{sketch}/{file}"
#                 sketchs_path.append(path)

#     return images_path,sketchs_path#,label_path


# ------------------------------------
# step 2 : 得到可迭代文件路径
# input : images_path,sketchs_path,label_path
# output : dict,value值为每张图片的路径
def iterate_path(images_path,sketchs_path,label_path):
    num = len(images_path)
    data_dict = {}
    for i in range(num):
        data_dict[i] = [images_path[i],sketchs_path[i],label_path[i]]
    return data_dict

# 简易版 dataloader，同时迭代image_path和sketch_path，没有label
# input ： images_path,sketchs_path type = List
# output ：data_dict,type = Dict
# def iterate_path(images_path,sketchs_path):
#     num = len(images_path)
#     data_dict = {}
#     for i in range(num):
#         data_dict[i] = [images_path[i],sketchs_path[i]]
#     return data_dict


# ------------------------------------
# step3 : 训练数据迭代图片路径，得到特征和groundtruth，并且上采样,将特征reshape
# input ： data_dict
# output ： feature 
def data_iterate(data_dict:dict,blocks,steps,dim,text_path):
    feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
    with open(text_path,"r",encoding='utf-8') as fp:
        text_content = json.load(fp)
    X = torch.zeros((len(data_dict),*dim[::-1]), dtype=torch.float) # [25,8640,81,256,256]
    Y = torch.zeros((len(data_dict), *dim[:-1]), dtype=torch.uint8) 
    for i in data_dict:
        x_0_path,sketch_path = data_dict[i][0],data_dict[i][1]
        text_name = x_0_path.split("/")[-1].split(".")[0]
        text_name = text_name+".jpg"
        label_route = data_dict[i][2]
        xc = text_content[text_name]
        # import pdb
        # pdb.set_trace()
        x_0 = cv2.imread(x_0_path)
        sketch = cv2.imread(sketch_path)
        activations = feature_extractor(x_0, sketch,xc)
        resize_activations = collect_features(256,activations,mode = "bilinear") # [8640,81, 256, 256]
        X[i] = resize_activations # should be [8640,81,256,256]
        label = np.load(label_route).astype('uint8')
        tensor_label = torch.from_numpy(label) # [256,256]
        Y[i] = tensor_label
        # import pdb
        # pdb.set_trace()
    d = X.shape[1]
    X = X.permute(1,0,2,3).reshape(d,-1).permute(1,0) #[256*256*25,8640,81]
    Y = Y.flatten() # [256*256*25] 1/0
    fp.close()
    return X,Y

def data_iterate_notext(data_dict:dict,blocks,steps,dim):
    feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
    X = torch.zeros((len(data_dict),*dim[::-1]), dtype=torch.float) # [num,8640,81,256,256]
    Y = torch.zeros((len(data_dict), *dim[:-1]), dtype=torch.uint8)
    for i in data_dict:
        x_0_path,sketch_path = data_dict[i][0],data_dict[i][1]
        text_name = x_0_path.split("/")[-1].split(".")[0]
        text_name = text_name+".png"
        label_route = data_dict[i][2]
        xc = ""
        x_0 = cv2.imread(x_0_path)
        sketch = cv2.imread(sketch_path)
        activations = feature_extractor(x_0,sketch,xc)
        resize_activations = collect_features(256,activations,mode = "bilinear") # [8640,81, 256, 256]
        X[i] = resize_activations # should be [8640,81,256,256]
        label = np.load(label_route).astype('uint8')
        tensor_label = torch.from_numpy(label) # [256,256]
        Y[i] = tensor_label
    # import pdb
    # pdb.set_trace()
    d = X.shape[1]
    X = X.permute(1,0,2,3).reshape(d,-1).permute(1,0) #[256*256*25,8640,81]
    Y = Y.flatten() # [256*256*25] 1/0
    return X,Y

# ------------------------------------------------------
# 测试集dataloader
# imput ： data_dict，blocks，steps，text_path
# output : test_feature_label_dict
def test_data_iterate_notext(data_dict:dict,blocks,steps,dim):
    feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
    # with open(text_path,"r",encoding='utf-8') as fp:
    #     text_content = json.load(fp)
    test_feature_label_dict = {}
    X = torch.zeros((len(data_dict),*dim[::-1]), dtype=torch.float) # [5,8640,81,256,256]
    for i in data_dict:
        x_0_path,sketch_path = data_dict[i][0],data_dict[i][1]
        text_name = x_0_path.split("/")[-1].split(".")[0]
        text_name = text_name+".png"
        label_route = data_dict[i][2]
        # xc = text_content[text_name]
        # import pdb
        # pdb.set_trace()
        x_0 = cv2.imread(x_0_path)
        sketch = cv2.imread(sketch_path)
        # sketch = np.random.randint(0,255,size=(256,256,3),dtype='uint8')
        # import pdb
        # pdb.set_trace()

        activations = feature_extractor(x_0, sketch,xc ="")
        resize_activations = collect_features(256,activations,mode = "bilinear") # [8640,81, 256, 256]
        X[i] = resize_activations
        label_route = data_dict[i][2]
        label = np.load(label_route).astype('uint8')
        tensor_label = torch.from_numpy(label)
        test_feature_label_dict[i] = [tensor_label,x_0_path]
        
    # fp.close()
    return test_feature_label_dict,X

# 测试集dataloader
# imput ： data_dict，blocks，steps，text_path
# output : test_feature_label_dict
def test_data_iterate(data_dict:dict,blocks,steps,dim,text_path):
    feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
    with open(text_path,"r",encoding='utf-8') as fp:
        text_content = json.load(fp)
    test_feature_label_dict = {}
    X = torch.zeros((len(data_dict),*dim[::-1]), dtype=torch.float) # [5,8640,81,256,256]
    for i in data_dict:
        x_0_path,sketch_path = data_dict[i][0],data_dict[i][1]
        text_name = x_0_path.split("/")[-1].split(".")[0]
        text_name = text_name+".jpg"
        label_route = data_dict[i][2]
        xc = text_content[text_name]
        # import pdb
        # pdb.set_trace()
        x_0 = cv2.imread(x_0_path)
        sketch = cv2.imread(sketch_path)

        activations = feature_extractor(x_0, sketch,xc)
        resize_activations = collect_features(256,activations,mode = "bilinear") # [8640,81, 256, 256]
        X[i] = resize_activations
        label_route = data_dict[i][2]
        label = np.load(label_route).astype('uint8')
        tensor_label = torch.from_numpy(label)
        test_feature_label_dict[i] = [tensor_label,x_0_path]
        
    # fp.close()
    return test_feature_label_dict,X

# 没有label的版本

# def test_data_iterate(data_dict:dict,blocks,steps,text_path):
#     feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
#     with open(text_path,"r",encoding='utf-8') as fp:
#         text_content = json.load(fp)
#     test_feature_label_dict = {}
#     for i in data_dict:
#         x_0_path,sketch_path = data_dict[i][0],data_dict[i][1]
#         text_name = x_0_path.split("/")[-1].split(".")[0]
#         text_name = text_name+".jpg"  # 根据text调整
#         xc = text_content[text_name]
#         x_0 = cv2.imread(x_0_path)
#         sketch = cv2.imread(sketch_path)
#         activations = feature_extractor(x_0, sketch,xc)
#         resize_activations = collect_features(256,activations,mode = "bilinear") # [8640,81, 256, 256]
#         test_feature_label_dict[i] = [resize_activations,x_0_path]
    # return test_feature_label_dict
    

class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_data: torch.Tensor, 
        y_data: torch.Tensor
    ):    
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

# if __name__ == "__main__":
    # image_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/photo"
    # sketch = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/sketch"
    # anno_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/anno_npy"

    # image_path = "/root/code_dir/ControlNet_Seg/datasets/instance_zero_test_choosed/photo/cat"
    # sketch = "/root/code_dir/ControlNet_Seg/datasets/instance_zero_test_choosed/sketch/cat"
    # anno_path = "/root/code_dir/ControlNet_Seg/datasets/instance_zero_test_choosed/anno_npy/cat"
    # images_path,sketchs_path,label_path = get_one_category_file(image_path,sketch,anno_path)
    
   

