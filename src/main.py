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
from tools import cal_metrics, cal_metrics_sperate

from models.cnn.timesnet import Timesnet
from models.cnn.modernTCN import ModernTCN

from models.mlp.mlp import MLP
from models.cnn.tcn import TCN
from models.gnn.gcn import GCN, generate_edge_weights
from models.rnn.lstm import LSTM

import itertools
import sys
from pprint import pprint
import warnings

warnings.filterwarnings('ignore')

# one batch call cal_metric one time 
def evaluation_sep(model, dataloader,dataset,device):

    model.eval()
    mean = dataset.all_timeseries_std_mean['WATER']['mean']
    std = dataset.all_timeseries_std_mean['WATER']['std']

    percent_10 = (dataset.percentile_mask_10['WATER'] - mean )/std
    percent_5 = (dataset.percentile_mask_5['WATER'] - mean )/std
    percent_1 = (dataset.percentile_mask_1['WATER'] - mean )/std

    all_metric = {}
    all_count = 0
    for batch in tqdm(dataloader):

        all_input = []
        all_output = []
        water_start = -1
        water_end = -1
        count = 0

        for key in ['WATER_input', 'RAIN_input', 'WELL_input', 'PUMP_input', 'GATE_input']:
            if key not in batch.keys():
                continue
            all_input.append(batch[key])
            all_output.append(batch[key.replace('_input', '_output')])
            if 'water' in key.lower():
                water_start = count
                water_end = count + batch[key].shape[1]
            count += batch[key].shape[1]

        input = torch.concat(all_input, dim=1)
        output = torch.concat(all_output, dim=1)

        pred = model(input.to(device)).detach()

        if pred.shape[1] == input.shape[1]:
            metrics = cal_metrics_sperate(output.cpu()[:, water_start:water_end, :],
                                  pred.cpu()[:, water_start:water_end, :].view(input.shape[0], -1, output.shape[-1]),
                                  mean, std, [percent_10, percent_5, percent_1])
        else:
            metrics = cal_metrics_sperate(output.cpu()[:, water_start:water_end, :],
                                  pred.cpu().view(input.shape[0], -1, output.shape[-1]),
                                  mean, std, [percent_10, percent_5, percent_1])

        for key in metrics.keys():
            if key in all_metric.keys():
                if 'sedi' in key:
                    all_metric[key][0] += metrics[key][0]
                    all_metric[key][1] += metrics[key][1]
                    all_metric[key][2] += metrics[key][2]
                else:
                    all_metric[key] += metrics[key]
            else:
                all_metric[key] = metrics[key]
        all_count += 1

    for key in all_metric.keys():
        if 'sedi' in key:
            all_metric[key][0] = (all_metric[key][0][:,:,0] / (all_metric[key][0][:,:,1]+1E-4)).mean()
            all_metric[key][1] = (all_metric[key][1][:,:,0] / (all_metric[key][1][:,:,1]+1E-4)).mean()
            all_metric[key][2] = (all_metric[key][2][:,:,0] / (all_metric[key][2][:,:,1]+1E-4)).mean()
        else:
            all_metric[key] = all_metric[key]/all_count
    pprint(all_metric)
    return all_metric

# all data call cal_metric one time 
def evaluation(model, dataloader,dataset,device):

    model.eval()
    mean = dataset.all_timeseries_std_mean['WATER']['mean']
    std = dataset.all_timeseries_std_mean['WATER']['std']

    outputs_water = []
    preds_water = []

    percent_10 = (dataset.percentile_mask_10['WATER'] - mean )/std
    percent_5 = (dataset.percentile_mask_5['WATER'] - mean )/std
    percent_1 = (dataset.percentile_mask_1['WATER'] - mean )/std

    all_metric = {}
    for batch in tqdm(dataloader):

        all_input = []
        all_output = []
        water_start = -1
        water_end = -1
        count = 0

        for key in ['WATER_input', 'RAIN_input', 'WELL_input', 'PUMP_input', 'GATE_input']:
            if key not in batch.keys():
                continue

            all_input.append(batch[key])
            all_output.append(batch[key.replace('_input', '_output')])
            if 'water' in key.lower():
                water_start = count
                water_end = count + batch[key].shape[1]
            count += batch[key].shape[1]

        input = torch.concat(all_input, dim=1)
        output = torch.concat(all_output, dim=1)

        pred = model(input.to(device)).detach()

        outputs_water.append(output.cpu()[:,water_start:water_end,:])
        preds_water.append(pred.cpu()[:,water_start:water_end,:].view(input.shape[0],-1,output.shape[-1]))

    outputs_water =  torch.concat(outputs_water,dim=0)
    preds_water =  torch.concat(preds_water,dim=0)
    all_metric = cal_metrics(outputs_water,
                              preds_water,
                              mean, std, [percent_10, percent_5, percent_1])
    pprint(all_metric)
    return all_metric

# only cacluate mse once 
def eval(model, dataloader,dataset,device):

    model.eval()

    all_metric = 0
    all_count = 0

    mse = nn.MSELoss()
    for batch in tqdm(dataloader):

        all_input = []
        all_output = []
        water_start = -1
        water_end = -1
        count = 0

        for key in ['WATER_input', 'RAIN_input', 'WELL_input', 'PUMP_input', 'GATE_input']:
            if key not in batch.keys():
                continue

            all_input.append(batch[key])
            all_output.append(batch[key.replace('_input', '_output')])
            if 'water' in key.lower():
                water_start = count
                water_end = count + batch[key].shape[1]
            count += batch[key].shape[1]

        input = torch.concat(all_input, dim=1)
        output = torch.concat(all_output, dim=1)

        pred = model(input.to(device)).detach()

        if pred.shape[1] == input.shape[1]:
            metrics = mse(output.cpu()[:, water_start:water_end, :],
                                  pred.cpu()[:, water_start:water_end, :].view(input.shape[0], -1, output.shape[-1]))
        else:
            metrics = mse(output.cpu()[:, water_start:water_end, :],
                                  pred.cpu().view(input.shape[0], -1, output.shape[-1]))

        all_metric += metrics
        all_count += 1

    return all_metric/all_count


def main(args):
    
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
    learning_rate = args.lr
    weight_decay = args.weight_decay
    epoches = args.epoches
    method_name = args.method
    cache_dir = args.cache_dir
    store = args.store

    store_dir = os.path.join(cache_dir,f'{method_name}_{dataset}_{length_input}_{length_output}')
    if store and not os.path.exists(store_dir):
        os.makedirs(store_dir)
    store_path = os.path.join(store_dir,"model_state.pth")
    # load dataset
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir,dataset,
                                length_input,length_span,length_output,batch_size,cache_dir=cache_dir,device=device)

    input_t_length = dataset_dict['train'].length_input
    span_t_length = dataset_dict['train'].length_span
    output_t_length = dataset_dict['train'].length_output

    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]

    # MLP
    if method_name.lower() == 'mlp'.lower():
        model = MLP(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    # CNN
    elif method_name.lower() == 'TCN'.lower():
        model = TCN(input_t_length,span_t_length,output_t_length,input_dim,output_dim,output_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
    elif method_name.lower() == 'GCN'.lower():
        all_locations = dataset_dict['train'].all_locations
        locations = [all_locations['WATER'],all_locations['RAIN'],all_locations['WELL'],all_locations['PUMP'],all_locations['GATE']]
        locations = np.concatenate(locations,axis=1)
        edge_index, edge_weight = generate_edge_weights(locations.T)
        model = GCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim,edge_index,edge_weights=edge_weight)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    elif method_name.lower() == 'LSTM'.lower():
        model = LSTM(input_t_length, span_t_length, output_t_length, input_dim, output_dim, output_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    else:
        raise ValueError("method error")

    model.to(device)

    train_dataloder = dataloader_dict['train']
    val_dataloder = dataloader_dict['val']
    test_dataloder = dataloader_dict['test']


    criterion = nn.MSELoss()
    best_eval = float('inf')
    best_model_dict = model.state_dict()
    for epoch in range(epoches):
        sum_loss = 0
        count_loss = 0
        for batch in tqdm(train_dataloder):

            all_input = []
            all_input_mask = []
            all_output = []
            all_output_mask = []
            water_start = -1
            water_end = -1
            count = 0
            for key in batch.keys():

                if "_input_mask" in key:
                    all_input_mask.append(batch[key])
                elif "_input" in key:
                    all_input.append(batch[key])
                    if 'water' in key.lower():
                        water_start = count
                        water_end = count + batch[key].shape[1]
                    count += batch[key].shape[1]
                elif "_output_mask" in key:
                    all_output_mask.append(batch[key])
                elif "_output" in key:
                    all_output.append(batch[key])

            input = torch.concat(all_input,dim=1)
            input_mask = torch.concat(all_input_mask,dim=1)
            output = torch.concat(all_output,dim=1)
            output_mask = torch.concat(all_output_mask,dim=1)

            pred = model(input.to(device))
            loss = criterion(pred[:,water_start:water_end,:], output[:,water_start:water_end,:].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            count_loss += 1
            
        print(f'{epoch} loss:{sum_loss/count_loss}')
        metric_dict = eval(model,val_dataloder,dataset_dict['val'],device)
        if metric_dict<best_eval:
            best_eval = metric_dict
            best_model_dict = model.state_dict()

    model.load_state_dict(best_model_dict)
    # test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device,batch_size,valid=False)
    test_metric_dict = evaluation_sep(model,test_dataloder,dataset_dict['test'],device)
    for key in test_metric_dict.keys():
        print(f'{key}: {test_metric_dict[key]}')
    if store:
        torch.save(best_model_dict,store_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Dataset Benchmark')

    parser.add_argument('--dataset_path', default='../dataset/Processed_hour/')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--dataset', default='S_0', choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'], type=str)
    parser.add_argument('--length_input', default='3D', choices=['1D', '2D', '3D', '1W', '2W', '3W'], type=str)
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'], type=str)
    parser.add_argument('--length_output', default='1D', choices=['1H', '6H', '12H', '1D', '2D','3D','5D','1W'], type=str)

    parser.add_argument('--method', default='gcn', type=str)

    parser.add_argument('--lr', default=1E-3, type=float)
    parser.add_argument('--weight_decay', default=0E-5, type=float)
    parser.add_argument('--epoches', default=1, type=int)
    parser.add_argument('--batchsize', default=8, type=int)

    parser.add_argument('--store', default=True, type=bool)

    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    print(args)
    main(args)


