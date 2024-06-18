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
        import pdb
        pdb.set_trace()
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
            # eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context[0],control=control_, only_mid_control = False)
            # eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context_random,control=control_, only_mid_control = False)
            eps = self.model.model.diffusion_model(x=z_noise,timesteps = t,context = context[0],control=None, only_mid_control = False)
            
            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations[:,0:-1:2,:,:])
                # activations.append(block.activations)
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


if __name__=="__main__":
    blocks = [5,7,8,11]
    steps = [50,150,250]
    # import pdb
    # pdb.set_trace()
    feature_extractor =  FeatureExtractorDDPM(blocks=blocks,steps=steps)

    x_0 = cv2.imread("/root/code_dir/ControlNet_Seg/datasets/25_train_data/image/n01530575_1534.jpg")
    sketch = cv2.imread("/root/code_dir/ControlNet_Seg/datasets/25_train_data/sketch/n01530575_1534-2.png")

    activation = feature_extractor(x_0, sketch,xc="") # __call__ self.forward() pytorch forward __call__
    resize_activation = collect_features(256,activation,mode="bilinear") # [8640, 256, 256] 
