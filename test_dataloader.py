import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from get_feature import FeatureExtractorDDPM, collect_features
import cv2
from tqdm import tqdm

from pre_data import *

class Valid_DataSet(Dataset):
    def __init__(self,
                 images_path:list,
                 sketchs_path:list,
                 label_path:list) -> None:
        super().__init__()
        self.images_list = images_path
        self.sketchs_list = sketchs_path
        self.label_list = label_path
        blocks = [5,7,8,11]
        steps = [50,150,250]
        self.dim = [256,256,8640]
        self.feature_extractor = FeatureExtractorDDPM(blocks=blocks,steps=steps)
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index) :
        # import pdb
        # pdb.set_trace()
        image_path = self.images_list[index]
        sketch_path = self.sketchs_list[index]
        label_path = self.label_list[index]
        image = cv2.imread(image_path)
        sketch = cv2.imread(sketch_path)
        activations = self.feature_extractor(image, sketch,xc = [""])
        resize_activations = collect_features(256,activations,mode = "bilinear") # [8640, 256, 256]
        label = np.load(label_path).astype('uint8')
        tensor_label = torch.from_numpy(label)

        return tensor_label,image_path,resize_activations


if __name__ =="__main__":
    image_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/photo"
    sketch = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/sketch"
    anno_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/anno_npy"
    images_path,sketchs_path,label_path = get_file_list(image_path,sketch,anno_path)

    data = Valid_DataSet(images_path,sketchs_path,label_path)
    valid_dataloder = DataLoader(data,batch_size=5,drop_last=True)
    for (tensor_label,image_path,resize_activations) in tqdm(valid_dataloder):
        for i,_ in enumerate(image_path):
            import pdb
            pdb.set_trace()
            print(image_path[i])
            print(resize_activations.shape)