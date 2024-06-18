import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import time
from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform

from pre_data import *
from guided_diffusion.guided_diffusion.dist_util import dev


def prepare_data():
    image_path = "/root/code_dir/ControlNet_Seg/datasets/instance_level_train/photo"
    sketch = "/root/code_dir/ControlNet_Seg/datasets/instance_level_train/sketch"
    # text_path = "/root/code_dir/ControlNet_Seg/datasets/category_level_train/text/text.json"
    anno_path = "/root/code_dir/ControlNet_Seg/datasets/instance_level_train/anno_npy"
    blocks = [5,7,8,11]
    steps = [50,100,200]
    dim = [256,256,4320]
    images_path,sketchs_path,label_path = get_file_list(image_path,sketch,anno_path)
    # images_path,sketchs_path,label_path = get_one_category_file(image_path,sketch,anno_path)
    # images_path,sketchs_path,label_path = get_path(image_path,sketch)
    data = iterate_path(images_path,sketchs_path,label_path)
    feature,label = data_iterate_notext(data,blocks=blocks,steps=steps,dim=dim)
    # feature,label = data_iterate(data,blocks=blocks,steps=steps,dim=dim,text_path=text_path)
    return feature,label

def train(args):

    #-----------------------------------------------------
    # step 1 : prepare data
    pre_data_start =time.time()
    features, labels = prepare_data() 
    pre_data_end = time.time()
    pre_time = pre_data_end - pre_data_start
    print("********The time of data prepare is",f"{pre_time}","****")
    train_data = FeatureDataset(features, labels)
    print("-------------------data prepare over-----------------------")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    print("----------------data load over------------------------------")

    #--------------------------------------------------------
    # step 2 : train
    for MODEL_NUMBER in range(args['model_num']):
        gc.collect()
        # create model,model is a classifier
        classifier = pixel_classifier(numpy_class = 2, dim=4320)
        classifier.init_weights()
        # 多卡
        classifier = nn.DataParallel(classifier).cuda()
        # 注意 : 这里是数据并行 所以需要确定卡的数量 来判断batchsize 即4张卡 batchsize至少需要8
        # 单卡
        # classifier = classifier.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        print("--------------------training----------------------")
        for epoch in range(100):
            one_epoch_start = time.time()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev()) # X_batch.shape = [bz,8848]
                y_batch = y_batch.type(torch.long) # [bz]
                optimizer.zero_grad()
                y_pred = classifier(X_batch) # [bz,num_cls]
                loss = criterion(y_pred, y_batch) # 交叉熵
                acc = multi_acc(y_pred, y_batch) 
                loss.backward()
                optimizer.step()
                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > 4: # 可调参数
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50: # 可调参数
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break
            one_epoch_end = time.time()
            one_epoch_time = one_epoch_end - one_epoch_start
            print("The time of one epoch training is",f"{one_epoch_time}")
        #------------------------------------------------------
        # step 3 : save model
        model_path = os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--exp_dir',type=str,default="/root/code_dir/ControlNet_Seg/checkpoint/exp24")
    parser.add_argument('--model_num',type=int,default=5)
    parser.add_argument('--start_model_num',type=int,default=0)
    opts = {}
    args = parser.parse_args()
    setup_seed(args.seed)
    opts.update(vars(args))
    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)

