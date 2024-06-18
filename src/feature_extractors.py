import sys
import torch
from torch import nn
from typing import List



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


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


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        # import guided_diffusion.guided_diffusion.dist_util as dist_util
        
        # Needed to pass only expected args to the function
        # argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        # expected_args = {name: kwargs[name] for name in argnames}
        # self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        
        # self.model.load_state_dict(
        #     dist_util.load_state_dict(model_path, map_location="cpu")
        # )
        # self.model.to(dist_util.dev()) # 分布式
        # if kwargs['use_fp16']:
        #     self.model.convert_to_fp16()
        self.model.eval() # eval模式，不更新参数

    @torch.no_grad()
    def forward(self, x, noise=None):# x.shape=[1, 3, 256, 256] noise.shape=[1, 3, 256, 256]
        activations = []
        for t in self.steps:# self.steps=[50, 150, 250]

            # import pdb
            # pdb.set_trace()

            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device) 
            noisy_x = self.q_sample(x, t, noise=noise) # q{x_t|x_0} 
            self.model(noisy_x, self.diffusion._scale_timesteps(t)) # 返回epsilon_theta

            # Extract activations
            for block in self.feature_blocks:# len(self.feature_blocks) = 5
                activations.append(block.activations)
                block.activations = None
              
        # Per-layer list of activations [N, C, H, W] N = len(steps)*len(blocks)
        return activations 
        # activations [N, C, H, W], N = len(steps)*len(blocks)
        # activations[0].shape = [1, 1024, 32, 32]  activations[1:4].shape=torch.Size([1, 512, 32, 32])
        # activations[4].shape=[1, 256, 128, 128]
        # (1024 + 3*512 + 256) * 3 = 8448

def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])

    # import pdb
    # pdb.set_trace()
    size = tuple(args['dim'][:-1]) #[256,256]
    resized_activations = []
    for feats in activations: # feats.shape[1,C, H, W]
        feats = feats[sample_idx][None]  # [1,C,H, W]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"] # 把feature upsample到和数据大小相同 [1,C,256,256]
        )
        resized_activations.append(feats[0])  # feats[0].shape = [C,256,256]
    
    return torch.cat(resized_activations, dim=0) # [8448,256,256]
