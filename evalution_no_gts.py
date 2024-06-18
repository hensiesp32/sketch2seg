import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions_grt, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform

from guided_diffusion.guided_diffusion.dist_util import dev
from pre_data import test_data_iterate,iterate_path,get_path,FeatureDataset


# step 1 : data_prepare
def prepare_data():
    image_path = "/root/code_dir/ControlNet_Seg/datasets/5_test_data/image"
    sketch = "/root/code_dir/ControlNet_Seg/datasets/5_test_data/sketch"
    text_path = "/root/code_dir/ControlNet_Seg/datasets/5_test_data/text/simple_text.json"

    blocks = [5,7,8,11]
    steps = [50,150,250]
    dim = [256,256,8640]
    # images_path,sketchs_path,label_path = get_path(image_path,sketch)
    images_path,sketchs_path = get_path(image_path,sketch)
    # data = iterate_path(images_path,sketchs_path,label_path)
    data = iterate_path(images_path,sketchs_path)
    test_feature_label_dict = test_data_iterate(data,blocks=blocks,steps=steps,text_path=text_path) # 
    return test_feature_label_dict

def evaluation(args, models):
    # import pdb
    # pdb.set_trace()
    test_data_iterate = prepare_data()
    image_paths = []
    preds, gts, uncertainty_scores = [], [], []
    for key in test_data_iterate:        
        features = test_data_iterate[key][0]
        # label = test_data_iterate[key][1]
        # image_paths.append(test_data_iterate[key][2])
        image_paths.append(test_data_iterate[key][1])
        x = features.view(args['dim'][-1], -1).permute(1, 0) # [65535,8640]
    
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        # gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item()) # pixel_classifier.py predict_labels()的top_k
    
    save_predictions(args, image_paths, preds)
    # save_predictions_grt(args,image_paths, gts)

    # miou = compute_iou(args, preds, gts)
    # print(f'Overall mIoU: ', miou)
    # print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--exp_dir',type=str,default="/root/code_dir/ControlNet_Seg/save_text_train_dir318")
    parser.add_argument('--save_dir',type=str,default="/root/code_dir/ControlNet_Seg/exp_result/exp_5_fscoco")
    parser.add_argument('--model_num',type=int,default=3)
    parser.add_argument('--start_model_num',type=int,default=0)
    parser.add_argument('--dim',type=list,default=[256,256,8640])
    parser.add_argument('--ignore_label',type=int,default=0)
    opts = {}
    args = parser.parse_args()
    setup_seed(args.seed)
    opts.update(vars(args))

    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)