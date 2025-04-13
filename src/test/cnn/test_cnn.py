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
from dataloader import load_dataset_loader,sort_files
from evaluation import cal_metrics

from models.cnn.timesnet import Timesnet
from models.cnn.modernTCN import ModernTCN
from models.cnn.tcn import TCN

from models.mlp.nbeats import NBeats
from models.mlp.nlinear import NLinear
from models.mlp.tsmixer import TSMixer
from models.mlp.mlp import MLP

from models.llm.timellm import TimeLLM
from models.llm.s2ipllm import S2IPLLM

from models.rnn.deepAR import DeepAR
from models.rnn.dilateRNN import DilatedRNN

from models.gnn.stemGNN import stemGNN
from models.gnn.fourierGNN import FourierGNN
from models.gnn.gcn import GCN, generate_edge_weights

from models.transformer.autoformer import AutoFormer
from models.transformer.informer import Informer
from models.transformer.itransformer import iTransformer
from models.transformer.patchTST import PatchTST
from models.transformer.pedformer import FEDFormer

import itertools
import sys
from pprint import pprint
import warnings
import json


warnings.filterwarnings('ignore')


def evaluation(model, dataloader,dataset,device,batch_size = 32,valid=True):

    model.eval()
    mean = dataset.all_timeseries_std_mean['WATER']['mean']
    std = dataset.all_timeseries_std_mean['WATER']['std']

    # inputs_water = []
    outputs_water = []
    preds_water = []
    # percentile_mask_lists = [[],[],[]]


    percent_10 = (dataset.percentile_mask_10['WATER'] - mean )/std
    percent_5 = (dataset.percentile_mask_5['WATER'] - mean )/std
    percent_1 = (dataset.percentile_mask_1['WATER'] - mean )/std

    all_metric = {}
    all_count = 0
    for batch in tqdm(dataloader):

        all_input = []
        all_input_mask = []
        all_output = []
        all_output_mask = []
        water_start = -1
        water_end = -1
        count = 0

        for key in ['WATER_input', 'RAIN_input', 'WELL_input', 'PUMP_input', 'GATE_input']:
            all_input.append(batch[key])
            all_output.append(batch[key.replace('_input', '_output')])
            if 'water' in key.lower():
                water_start = count
                water_end = count + batch[key].shape[1]
            count += batch[key].shape[1]

        input = torch.concat(all_input, dim=1)
        output = torch.concat(all_output, dim=1)

        pred = model(input.to(device)).detach()

        # outputs_water.append(output.cpu()[:,water_start:water_end,:])
        # preds_water.append(pred.cpu()[:,water_start:water_end,:].view(input.shape[0],-1,output.shape[-1]))

        metrics = cal_metrics(output[:,water_start:water_end,:].cpu(), pred[:,water_start:water_end,:].cpu().view(input.shape[0],-1,output.shape[-1]),
                              mean,std,[percent_10, percent_5, percent_1])

        # if metrics['mse']>10:
        #     print('xxx')

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
            all_metric[key][0] = all_metric[key][0] /all_count
            all_metric[key][1] = all_metric[key][1] /all_count
            all_metric[key][2] = all_metric[key][2] /all_count
        else:
            all_metric[key] = all_metric[key]/all_count
    pprint(all_metric)
    return all_metric

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
    learning_rate = args.lr
    weight_decay = args.weight_decay
    epoches = args.epoches
    method_name = args.method
    cache_dir = args.cache_dir
    # load dataset
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir,dataset,
                                length_input,length_span,length_output,batch_size,cache_dir=cache_dir,device=device,
                                                        use_cache=False)

    input_t_length = dataset_dict['train'].length_input
    span_t_length = dataset_dict['train'].length_span
    output_t_length = dataset_dict['train'].length_output

    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        if dataset_dict['train'].all_timeseries[key] is None:
            continue
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]

    # MLP
    if method_name.lower() == 'nbeats'.lower():
        model = NBeats(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'NLinear'.lower():
        model = NLinear(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'TSMixer'.lower():
        model = TSMixer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'mlp'.lower():
        model = MLP(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    # LLM
    elif method_name.lower() == 'timellm'.lower():
        model = TimeLLM(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)

        # parameters = [model.patch_embedding,model.output_projection]
        params = itertools.chain(model.patch_embedding.parameters(), model.output_projection.parameters())
        optimizer = optim.AdamW(params, lr=learning_rate,
                               weight_decay=weight_decay)

    elif method_name.lower() == 'S2IPLLM'.lower():
        model = S2IPLLM(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    # RNN
    elif method_name.lower() == 'DeepAR'.lower():
        model = DeepAR(input_t_length, span_t_length, output_t_length, output_dim, output_dim, output_dim)
    elif method_name.lower() == 'DilatedRNN'.lower():
        model = DilatedRNN(input_t_length, span_t_length, output_t_length, output_dim, output_dim, output_dim)

    # GNN
    elif method_name.lower() == 'stemGNN'.lower():
        model = stemGNN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    elif method_name.lower() == 'FourierGNN'.lower():
        model = FourierGNN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
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

    # CNN
    elif method_name.lower() == 'ModernTCN'.lower():
        model = ModernTCN(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'TCN'.lower():

        model = TCN(input_t_length,span_t_length,output_t_length,input_dim,input_dim,input_dim)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)

    elif method_name.lower() == 'Timesnet'.lower():
        model = Timesnet(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)

    # Transformer
    elif method_name.lower() == 'FEDFormer'.lower():
        model = FEDFormer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'PatchTST'.lower():
        model = PatchTST(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'iTransformer'.lower():
        model = iTransformer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'Informer'.lower():
        model = Informer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'AutoFormer'.lower():
        model = AutoFormer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)

    else:
        raise ValueError("method error")

    model.to(device)

    # train
    # epoches =  1
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
            # all_input_mask = []
            all_output = []
            # all_output_mask = []
            water_start = -1
            water_end = -1
            count = 0
            for key in ['WATER_input','RAIN_input','WELL_input','PUMP_input','GATE_input']:
                all_input.append(batch[key])
                all_output.append(batch[key.replace('_input','_output')])
                if 'water' in key.lower():
                    water_start = count
                    water_end = count + batch[key].shape[1]
                count += batch[key].shape[1]

            input = torch.concat(all_input,dim=1)
            output = torch.concat(all_output,dim=1)

            pred = model(input.to(device))
            loss = criterion(pred[:,water_start:water_end,:], output[:,water_start:water_end,:].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            count_loss += 1
        print(f'{epoch} loss:{sum_loss/count_loss}')
        # test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device,batch_size)
        metric_dict = evaluation(model,val_dataloder,dataset_dict['val'],device,batch_size)
        if metric_dict['mse']<best_eval:
            best_eval = metric_dict['mse']
            best_model_dict = model.state_dict()

    model.load_state_dict(best_model_dict)
    test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device,batch_size)
    for key in test_metric_dict.keys():
        print(f'{key}: {test_metric_dict[key]}')


if __name__=="__main__":
    print("Number of arguments:", len(sys.argv))
    print("Arguments are:", str(sys.argv))
    for i, arg in enumerate(sys.argv):
        print(f" {arg} ")

    parser = argparse.ArgumentParser(
        prog='Dataset Benchmark',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--dataset_path', default='../dataset/Processed')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--dataset', default='S_0', choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'])
    parser.add_argument('--length_input', default='3D', choices=['1D', '2D', '3D', '1W', '2W', '3W'])
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'])
    parser.add_argument('--length_output', default='12H', choices=['1H', '6H', '12H', '1D', '2D'])

    parser.add_argument('--method', default='TCN') # S2IPLLM

    parser.add_argument('--lr', default=5E-4)
    parser.add_argument('--weight_decay', default=1E-6)
    parser.add_argument('--epoches', default=10)
    parser.add_argument('--batchsize', default=256)


    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    print(args)
    main(args)


"""
Namespace(dataset_path='../dataset/Processed', cache_dir='./cache', dataset='S_0', length_input='3D', length_span='0H', length_output='12H', method='TCN', lr=0.0005, weight_decay=1e-06, epoches=5, batchsize=256, device='cuda:0', seed=2025)
mae: 0.5952771902084351
mse: 0.9157943725585938
rmse: 0.9081471562385559
mape: 4.826462268829346
mspe: 6868399.0
sedi: [np.float64(0.023409292746885236), np.float64(0.009810948993636529), np.float64(0.0016704894000685055)]
nse: -354.00830078125
"""

