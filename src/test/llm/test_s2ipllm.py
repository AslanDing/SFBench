import os
import torch
import torch.nn as nn
import numpy as np
import argparse

import datetime
import logging
import sys
import random
import torch.optim as optim
from tqdm import tqdm
from dataloader import load_dataset_loader
from evaluation import cal_metrics
import itertools
from S2IPLLM.s2ipllm import S2IPLLM
import warnings
warnings.filterwarnings('ignore')

def evaluation(model, dataloader,dataset,device):

    model.eval()
    mean = dataset.all_timeseries_std_mean['WATER']['mean']
    std = dataset.all_timeseries_std_mean['WATER']['std']

    inputs_water = []
    outputs_water = []
    preds_water = []
    # percentile_mask_lists = [[],[],[]]
    for batch in dataloader:

        input = batch['WATER_input']  # B,N,T
        output = batch['WATER_output']  # B,N,T
        pred = model(input.view(-1, input.shape[-1]).to(device))

        inputs_water.append(input.cpu())
        outputs_water.append(output.cpu())
        preds_water.append(pred.cpu().view(input.shape[0],-1,output.shape[-1]))
    inputs_water = torch.concat(inputs_water,dim=0)
    outputs_water = torch.concat(outputs_water,dim=0)
    preds_water = torch.concat(preds_water,dim=0)

    percent_10 = (dataset.percentile_mask_10['WATER'] - mean )/std
    percent_5 = (dataset.percentile_mask_5['WATER'] - mean )/std
    percent_1 = (dataset.percentile_mask_1['WATER'] - mean )/std

    # percentile_mask_lists[0] = torch.concat(percentile_mask_lists[0])
    # percentile_mask_lists[1] = torch.concat(percentile_mask_lists[1])
    # percentile_mask_lists[2] = torch.concat(percentile_mask_lists[2])
    metrics = cal_metrics(inputs_water,preds_water,outputs_water,mean,std,[percent_10,percent_5,percent_1])
    return metrics


def main(args):

    # data_dir, dataset, method, device, SEED
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True


    data_dir = args.dataset_path
    dataset = args.dataset
    length_input = args.length_input
    length_span = args.length_span
    length_output = args.length_output
    batch_size = args.batchsize
    device = args.device
    # load dataset
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir,dataset,
                                length_input,length_span,length_output,batch_size)

    input_t_length = dataset_dict['train'].length_input
    span_t_length = dataset_dict['train'].length_span
    output_t_length = dataset_dict['train'].length_output

    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]

    model = S2IPLLM(input_t_length,span_t_length,output_t_length,input_dim,input_dim,output_dim
                    )

    model.to(device)
    # train
    epoches =  1
    train_dataloder = dataloader_dict['train']
    val_dataloder = dataloader_dict['val']
    test_dataloder = dataloader_dict['test']

    # parameters = [model.patch_embedding,model.output_projection]
    # params = itertools.chain(model.patch_embedding.parameters(), model.output_projection.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=1e-4,
                           weight_decay=0e-5)
    criterion = nn.MSELoss()
    best_eval = float('inf')
    best_model_dict = model.state_dict()
    for epoch in tqdm(range(epoches)):
        sum_loss = 0
        count_loss = 0
        for batch in train_dataloder:
            # all_input = []
            # for key in batch.keys():
            #     if "_input" in key:
            #         all_input.append(batch[key])


            input = batch['WATER_input']  # B,N,T
            # input = torch.concat(all_input,dim=1)
            output = batch['WATER_output'] # B,N,T

            pred = model(input.to(device))
            loss = criterion(pred,output.view(-1,output.shape[-1]).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            count_loss += 1
        print(f'{epoch} loss:{sum_loss/count_loss}')
        metric_dict = evaluation(model,val_dataloder,dataset_dict['val'],device)
        if metric_dict['mse']<best_eval:
            best_eval = metric_dict['mse']
            best_model_dict = model.state_dict()

    model.load_state_dict(best_model_dict)
    test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device)
    for key in test_metric_dict.keys():
        print(f'{key}: {test_metric_dict[key]}')


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        prog='Dataset Benchmark',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--dataset_path', default='../dataset_download/Processed')
    parser.add_argument('--dataset', default='S_0', choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'])
    parser.add_argument('--length_input', default='1D', choices=['1D', '2D', '3D', '1W', '2W', '3W'])
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'])
    parser.add_argument('--length_output', default='1D', choices=['1H', '6H', '12H', '1D', '2D'])

    parser.add_argument('--method', default='mlp')

    parser.add_argument('--lr', default=1E-4)
    parser.add_argument('--batchsize', default=1)


    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    main(args)



