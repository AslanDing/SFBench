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
from dataloader import load_dataset_loader, load_dataset_part_loader
from tools import cal_metrics,cal_metrics_sperate

from models.mlp.mlp import MLP
from models.cnn.tcn import TCN
from models.gnn.gcn import GCN, generate_edge_weights
from models.rnn.lstm import LSTM

from models.cnn.timesnet import Timesnet
from models.cnn.modernTCN import ModernTCN

from models.mlp.nlinear import NLinear
from models.mlp.tsmixer import TSMixer

from models.rnn.deepAR import DeepAR
from models.rnn.dilateRNN import DilatedRNN

from models.gnn.stemGNN import stemGNN
from models.gnn.fourierGNN import FourierGNN

from models.transformer.itransformer import iTransformer
from models.transformer.patchTST import PatchTST

from models.llm.onefitall import GPT4TS
from models.llm.autotimes import AutoTimes


import itertools
import sys
from pprint import pprint
import warnings
import json


warnings.filterwarnings('ignore')



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
        all_input_mask = []
        all_output = []
        all_output_mask = []
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


def eval(model, dataloader,dataset,device):

    model.eval()

    all_metric = 0
    all_count = 0

    mse = nn.MSELoss()
    for batch in tqdm(dataloader):

        all_input = []
        all_input_mask = []
        all_output = []
        all_output_mask = []
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
    method_name = args.method
    cache_dir = args.cache_dir

    
    store_dir = os.path.join(cache_dir,f'{method_name}_{dataset}_{length_input}_{length_output}')
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
    # CNN
    elif method_name.lower() == 'TCN'.lower():
        model = TCN(input_t_length,span_t_length,output_t_length,input_dim,output_dim,output_dim)
    elif method_name.lower() == 'GCN'.lower():
        all_locations = dataset_dict['train'].all_locations
        locations = [all_locations['WATER'],all_locations['RAIN'],all_locations['WELL'],all_locations['PUMP'],all_locations['GATE']]
        locations = np.concatenate(locations,axis=1)
        edge_index, edge_weight = generate_edge_weights(locations.T)
        model = GCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim,edge_index,edge_weights=edge_weight)
    elif method_name.lower() == 'LSTM'.lower():
        model = LSTM(input_t_length, span_t_length, output_t_length, input_dim, output_dim, output_dim)
    else:
        raise ValueError("method error")
    
    best_model_dict = torch.load(store_path, map_location = 'cpu')
    model.load(best_model_dict)
    model.to(device)
    model.eval()

    train_dataloder = dataloader_dict['train']
    val_dataloder = dataloader_dict['val']
    test_dataloder = dataloader_dict['test']

    # test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device,batch_size,valid=False)
    test_metric_dict = evaluation_sep(model,test_dataloder,dataset_dict['test'],device,batch_size,valid=False)
    for key in test_metric_dict.keys():
        print(f'{key}: {test_metric_dict[key]}')

def main_parts(args):
    
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
    store_path = os.path.join(store_dir,"model_state_%d.pth")

    dataset_dict, dataloader_dict = load_dataset_part_loader(data_dir,dataset,
                                length_input,length_span,length_output,batch_size,cache_dir=cache_dir,device=device)

    nums = len(dataset_dict['train'])
    metrics_list = []
    for part_i in range(nums):

        input_t_length = dataset_dict['train'][part_i].length_input
        span_t_length = dataset_dict['train'][part_i].length_span
        output_t_length = dataset_dict['train'][part_i].length_output

        input_dim = 0
        for key in dataset_dict['train'][part_i].all_timeseries.keys():
            if dataset_dict['train'][part_i].all_timeseries[key] == None:
                continue
            dim_tmp = dataset_dict['train'][part_i].all_timeseries[key].shape[0]
            input_dim += dim_tmp
        output_dim = dataset_dict['train'][part_i].all_timeseries['WATER'].shape[0]

        # MLP
        if method_name.lower() == 'mlp'.lower():
            model = MLP(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'NLinear'.lower():
            model = NLinear(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'TSMixer'.lower():
            model = TSMixer(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        
        # LLM
        elif method_name.lower() == 'GPT4TS'.lower():
            model = GPT4TS(input_t_length, span_t_length, output_t_length, input_dim,
                           input_dim, input_dim,device=device)
        elif method_name.lower() == 'AutoTimes'.lower():
            model = AutoTimes(input_t_length, span_t_length, output_t_length, input_dim,
                           input_dim, input_dim,device=device)
        
        # RNN
        elif method_name.lower() == 'DeepAR'.lower():
            model = DeepAR(input_t_length, span_t_length, output_t_length, input_dim, input_dim, input_dim)
        elif method_name.lower() == 'DilatedRNN'.lower():
            model = DilatedRNN(input_t_length, span_t_length, output_t_length, input_dim, input_dim, input_dim)
        elif method_name.lower() == 'LSTM'.lower():
            model = LSTM(input_t_length, span_t_length, output_t_length, input_dim, input_dim, input_dim)

        # GNN
        elif method_name.lower() == 'stemGNN'.lower():
            model = stemGNN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'FourierGNN'.lower():
            model = FourierGNN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'GCN'.lower():
            all_locations = dataset_dict['train'][part_i].all_locations
            locations = []
            for location_tmp in [all_locations['WATER'],all_locations['RAIN'],all_locations['WELL'],all_locations['PUMP'],all_locations['GATE']]:
                if isinstance(location_tmp,np.ndarray) and location_tmp.shape[1]>=1:
                    locations.append(location_tmp)
            locations = np.concatenate(locations,axis=1)
            edge_index, edge_weight = generate_edge_weights(locations.T)
            model = GCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim,edge_index,
                        hidden_channels=32, num_hidden=3,
                        edge_weights=edge_weight)

        # CNN
        elif method_name.lower() == 'TCN'.lower():
            model = TCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'ModernTCN'.lower():
            model = ModernTCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'Timesnet'.lower():
            model = Timesnet(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)

        # Transformer
        elif method_name.lower() == 'PatchTST'.lower():
            model = PatchTST(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        elif method_name.lower() == 'iTransformer'.lower():
            model = iTransformer(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        else:
            raise ValueError("method error")

        best_model_dict = torch.load(store_path%part_i, map_location = 'cpu')
        model.load(best_model_dict)
        model.to(device)
        model.eval()


        train_dataloder = dataloader_dict['train'][part_i]
        val_dataloder = dataloader_dict['val'][part_i]
        test_dataloder = dataloader_dict['test'][part_i]

        test_metric_dict = evaluation_sep(model,test_dataloder,dataset_dict['test'][part_i],device,batch_size)
        metrics_list.append(test_metric_dict)
        for idx,metric_d in enumerate(metrics_list):
            print("part i : ", idx)
            for key in metric_d.keys():
                print(f'{key}: {metric_d[key]}')
        
        if store:
            torch.save(best_model_dict,store_path%part_i)

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Dataset Benchmark')

    parser.add_argument('--dataset_path', default='../dataset/Processed')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--dataset', default='S_2', choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'])
    parser.add_argument('--length_input', default='3D', choices=['1D', '2D', '3D', '1W', '2W', '3W'])
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'])
    parser.add_argument('--length_output', default='2D', choices=['1H', '6H', '12H', '1D', '2D'])

    parser.add_argument('--method', default='gcn')

    parser.add_argument('--store', default=False)
    parser.add_argument('--part', default=False)  # whole dataset or part

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    print(args)
    if args.part:
        main_parts(args)
    else:
        main(args)
