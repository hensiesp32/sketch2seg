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
from pre_data import *
from test_dataloader import Valid_DataSet

# step 1 : data_prepare
def prepare_data():
    # image_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/photo"
    # sketch = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/sketch"
    # anno_path = "/root/code_dir/ControlNet_Seg/datasets/few_shot_train_data/anno_npy"

    image_path = "/root/code_dir/ControlNet_Seg/datasets/PASCAL/pascal_voc_256/photo/motorcycle"
    sketch = "/root/code_dir/ControlNet_Seg/datasets/PASCAL/pascal_voc_256/sketch/motorcycle"
    anno_path = "/root/code_dir/ControlNet_Seg/datasets/PASCAL/pascal_voc_256/anno_npy/motorcycle"
    # text_path = "/root/code_dir/ControlNet_Seg/datasets/category_zero_test/text/text.json"

    blocks = [5,7,8,11]
    steps = [50,100,200]
    dim = [256,256,4320]
    # images_path,sketchs_path,label_path = get_path(image_path,sketch)
    # images_path,sketchs_path,label_path = get_file_list(image_path,sketch,anno_path)
    images_path,sketchs_path,label_path = get_one_category_file(image_path,sketch,anno_path)
    data = iterate_path(images_path,sketchs_path,label_path)

    test_feature_label_dict,X = test_data_iterate_notext(data,blocks=blocks,steps=steps,dim=dim) # 
    return test_feature_label_dict,X

def evaluation(args, models):
    # import pdb
    # pdb.set_trace()
    # step 1:数据准备
    test_data_iterate, X = prepare_data()
    image_paths = []
    preds, gts, uncertainty_scores = [], [], []


    for key in test_data_iterate:        
        # features = test_data_iterate[key][0]
        label = test_data_iterate[key][0]
        image_paths.append(test_data_iterate[key][1])
        # image_paths.append(test_data_iterate[key][1])
        x = X[key].view(args['dim'][-1], -1).permute(1, 0) # [65535,8640]

    # step 2 : reference
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item()) # pixel_classifier.py predict_labels()的top_k
    
    # step 3 : save result
    save_predictions(args, image_paths, preds)
    save_predictions_grt(args,image_paths, gts)

    miou,pixel_precision = compute_iou(args, preds, gts)
    print(f'Overall mIoU: ', miou)
    print(f'Overall pixel_precision: ',pixel_precision)
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')
    with open("/root/code_dir/ControlNet_Seg/exp_result/exp3_photo_only_voc/metric.txt","a") as fp:
        fp.write(f"Overall mIoU: {miou}\n")
        fp.write(f"Overall pixel_precision: {pixel_precision}\n")
        fp.write(f"Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}\n")
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--exp_dir',type=str,default="/root/code_dir/ControlNet_Seg/checkpoint/exp23")
    parser.add_argument('--save_dir',type=str,default="/root/code_dir/ControlNet_Seg/exp_result/exp3_photo_only_voc")
    parser.add_argument('--model_num',type=int,default=5)
    parser.add_argument('--start_model_num',type=int,default=0)
    parser.add_argument('--dim',type=list,default=[256,256,4320])
    parser.add_argument('--ignore_label',type=int,default=0)
    opts = {}
    args = parser.parse_args()
    setup_seed(args.seed)
    opts.update(vars(args))

    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)