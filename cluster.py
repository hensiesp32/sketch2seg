from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

# from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import sys
# import torch
from torch import nn
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module.
        任给一个模型，以及feature[]
        保存nerual_layer输入/输出的feature,还会给每一个feature一个名字
    """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractorDDPM(nn.Module):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    def __init__(self,blocks: List[int], steps:List[int],**kwargs):
        super().__init__(**kwargs)
        self._load_pretrained_model()
        # print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_out_hook
        self.feature_blocks = []
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.model.diffusion_model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self):
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
        self.model = self.model.cuda()
        self.model.eval() # eval模式，不更新参数，关闭dropout()

    @torch.no_grad() # 不累计梯度
    def forward(self, x_0, sketch,xc):
        '''
        得到feature
        input : 
            x_0 : [] mean tensor.float32
            sketch : [] mean list
        return :
            output : [] mean 
        '''
        # --------------------------------------------
        # step 1 : 将数据转换
        # input : x_0 sketch
        # output : source[B, 256, 256, C], target[B, 256, 256, C]
        activations = []
        source = cv2.cvtColor(x_0, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        source = source.astype(np.float32) / 255.0
        source = torch.from_numpy(source).view(1,256,256,3)
        target = (target.astype(np.float32) / 127.5) - 1.0
        target = torch.from_numpy(target).view(1,256,256,3)

        # for debug
        # import torchvision
        # torchvision.utils.save_image(source.permute(0, 3,1,2), 'test.png')
        # torchvision.utils.save_image(target.permute(0, 3,1,2), 'test.png')
        
        # --------------------------------------------
        # step 2 : get control
        # intput : source[B, 256, 256, C], target[B, 256, 256, C]
        # output : control_[]
        xc= [xc]
        # xc=[""]
        batch = {"jpg":source,"hint":target,"txt":xc}
        z,c = self.model.get_input(batch = batch,k = "jpg") # [4]
        control = c["c_concat"]
        context = c["c_crossattn"]
        # import pdb
        # pdb.set_trace()

        # context_random = torch.rand_like(context[0])
        for t in self.steps:
            t = torch.tensor([t]).to(z.device)
            control_ = self.model.control_model(x=z,timesteps = t,hint = control[0],context = context[0])
            # control_ = self.model.control_model(x=z,timesteps = t,hint = control[0],context = context_random)

            # --------------------------------------------
            # step 3 : get actvations
            noise = torch.randn_like(z)
            z_noise = self.model.q_sample(x_start = z,t=t,noise = noise)  # z_t
            eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context[0],control=control_, only_mid_control = False)
            # eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context_random,control=control_, only_mid_control = False)
            # eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context[0],control=None, only_mid_control = False)
            
            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None
        # import pdb
        # pdb.set_trace()
        return activations 
    """
    256*256
    activations[0].shape = [1, 1280, 16, 16]
    activations[1].shape = [1, 640, 16, 16]
    activations[2].shape = [1, 640, 32, 32]
    activations[3].shape = [1, 320, 32, 32]
    (1280+640*2+320)*3 = 8640

    512*512
    activations[0].shape = [1, 1280, 32, 32]
    activations[1].shape = [1, 640, 32, 32]
    activations[2].shape = [1, 640, 64, 64]
    activations[3].shape = [1, 320, 64, 64]
    (1280+640*2+320)*3 = 8640
    """
        
def collect_features(dim:int,activations,mode,sample_idx = 0):
    dim = tuple([dim,dim])
    resized_activations = []
    for feats in activations:
        feat = feats[sample_idx][None]
        feat = nn.functional.interpolate(feat, size=dim, mode=mode )# 把feature upsample到和数据大小相同 [1,C,256,256]
        resized_activations.append(feat[0])
    
    return torch.cat(resized_activations, dim=0)

def cluster(image_path,activations):
    # bgr_list = [(128, 128, 255), # 粉红色
    #              (128, 255, 128), # 亮绿色
    #              (255, 128, 128), # 紫色
    #              (128, 0, 255), # 玫红色
    #              (128, 255, 0), # 青绿色
    #              (255, 128, 0), # 蓝色
    #              (0, 128, 255), # 亮橙色
    #              (0, 255, 128), # 嫩绿色
    #              (255, 0, 128)] # 深紫色
    bgr_list = [(193, 182, 255), # 粉红色
                 (113, 179, 60), # 亮绿色
                 (238,104,123), # 紫色
                 (147,112,219), # 玫红色
                 (35,142,107), # 青绿色
                 (250,206,135), # 蓝色
                 (0,165,255), # 亮橙色
                 (87,139,46), # 嫩绿色
                 (205,90,106)] # 深紫色
    image = mpimg.imread(image_path)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(image)
    height, width, channel = activations.shape
    # do kmeans segmentation
    for i, k in enumerate(range(2,3,1)):
        # extract bgr and location features
        features = []
        for y in range(height):
            for x in range(width):
                features.append(np.concatenate((activations[y, x, :], np.array([y / height, x / width])), axis=0))
        features = np.array(features)
        # initial segments center using random value in features
        kmeans_centers = features[np.random.choice(len(features), k), :]
        kmeans_centers = np.array(kmeans_centers)
        # update
        while True:
            # calculate distance matrix
            def euclidean_dist(X, Y):
                Gx = np.matmul(X, X.T)
                Gy = np.matmul(Y, Y.T)
                diag_Gx = np.reshape(np.diag(Gx), (-1, 1))
                diag_Gy = np.reshape(np.diag(Gy), (-1, 1))
                return diag_Gx + diag_Gy.T - 2 * np.matmul(X, Y.T)
            dist_matrix = []
            for start in range(0, len(features), 1000):
                dist_matrix.append(euclidean_dist(features[start:start+1000, :], kmeans_centers))
            dist_matrix = np.concatenate(dist_matrix, axis=0)
            # dist_matrix = euclidean_dist(features, kmeans_centers)
            # get seg class for each sample
            segs = np.argmin(dist_matrix, axis=1)
            # update new kmeans center
            new_kmeans_centers = []
            for j in range(k):
                new_kmeans_centers.append(np.mean(features[segs==j, :], axis=0))
            new_kmeans_centers = np.array(new_kmeans_centers)
            # calculate whether converge
            if np.mean(abs(kmeans_centers - new_kmeans_centers)) < 0.1:
                break
            else:
                kmeans_centers = new_kmeans_centers
        # assign
        segs = segs.reshape(height, width)
        seg_result = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                seg_result[y, x, :] = bgr_list[segs[y, x]]
        # import pdb
        # pdb.set_trace()
        save_path = f"/root/code_dir/ControlNet_Seg/cluster_data/result_paper/n01887787_700_sketch_7.jpg"
        npy_save_path = f"/root/code_dir/ControlNet_Seg/cluster_data/result_npy/n01887787_700_sketch_7.npy"
        cv2.imwrite(save_path,seg_result)
        np.save(npy_save_path,seg_result)
        # show kmeans result
        plt.subplot(1, 2, i+2)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(seg_result)
        plt.savefig('/root/code_dir/ControlNet_Seg/cluster_data/result_paper/n01887787_700_sketch7.jpg')
    return 0

if __name__=="__main__":
    blocks = [5,7,8,11]
    steps = [50,150,250]
    # import pdb
    # pdb.set_trace()
    feature_extractor =  FeatureExtractorDDPM(blocks=blocks,steps=steps)
    image_path = "/root/code_dir/ControlNet_Seg/datasets/dataset/photo/cow/n01887787_700.jpg"
    sketch_path = "/root/code_dir/ControlNet_Seg/datasets/dataset/sketch/cow/n01887787_700-7.png"
    x_0 = cv2.imread(image_path)
    sketch = cv2.imread(sketch_path)

    activation = feature_extractor(x_0, sketch,xc = "") # __call__ self.forward() pytorch forward __call__
    resize_activation = collect_features(256,activation,mode="bilinear").permute(1,2,0).cpu() # [8640, 256, 256] 
    result = cluster(image_path,resize_activation)
