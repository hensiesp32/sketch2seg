import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        return self.layers(x)

# pixel 做预测
def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            preds = models[MODEL_NUMBER](features.cuda()) # [65525,2]
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds) # 对每一个像素分类[65535,2]
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds) # [65535]
            img_seg = img_seg.reshape(*size)  # [256,256]
            img_seg = img_seg.cpu().detach() 

            seg_mode_ensemble.append(img_seg)  # len(seg_mode_ensemble)=3

        # import pdb
        # pdb.set_trace()
        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0) # js.shape [65535]
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean() # top_k的含义？？

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1) #[256,256,5]
        img_seg_final = torch.mode(img_seg_final, 2)[0] # [256,256] img_seg_final[0][0:10] = [6, 6, 6, 6, 6, 6, 6, 6, 1, 1] 预测的类别
    return img_seg_final, top_k 
# 保存预测结果与可视化

def save_predictions(args, image_paths, preds):
    palette = get_palette()
    os.makedirs(os.path.join(args['save_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['save_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['save_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['save_dir'], 'visualizations', filename + '.jpg')
        )
# 可视化groundtruth
def save_predictions_grt(args, image_paths, gts):
    palette = get_palette()
    os.makedirs(os.path.join(args['save_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['save_dir'], 'visualizations'), exist_ok=True)

    for i, gts in enumerate(gts):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        gts = np.squeeze(gts)
        # np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), gts)

        mask = colorize_mask(gts, palette)
        Image.fromarray(mask).save(
            os.path.join(args['save_dir'], 'visualizations', filename + '_gts.jpg')
        )


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names()

    ids = range(2)

    unions = Counter()
    intersections = Counter()
    pixel_accs = []
    for pred, gt in zip(preds, gts):
        pixel_acc = (pred == gt).sum()/65536
        pixel_accs.append(pixel_acc)
        for target_num in ids:
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean(),np.array(pixel_accs).mean()


def load_ensemble(args, device='cpu'):
    # import pdb
    # pdb.set_trace()
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(numpy_class=2, dim=4320))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models
