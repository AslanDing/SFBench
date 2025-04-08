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

from models.cnn.timesnet import Timesnet
from models.cnn.modernTCN import ModernTCN

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

from models.transformer.autoformer import AutoFormer
from models.transformer.informer import Informer
from models.transformer.itransformer import iTransformer
from models.transformer.patchTST import PatchTST
from models.transformer.pedformer import FEDFormer

import itertools
import sys
import pprint
import warnings

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

        input = torch.concat(all_input, dim=1)
        input_mask = torch.concat(all_input_mask, dim=1)
        output = torch.concat(all_output, dim=1)
        output_mask = torch.concat(all_output_mask, dim=1)

        pred = model(input[:,water_start:water_end,:].to(device)).detach()

        # input = input[:, water_start:water_end, :].contiguous()
        # output = output[:, water_start:water_end, :].contiguous()
        # B, N, T = input.shape
        #
        # input = input.view(1, B * N, T)
        # output = output.view(1, B * N, -1)
        #
        # pred = []
        # for i in range(input.shape[1] // 32):
        #     input_tmp = input[:, i * batch_size:(i + 1) * batch_size]
        #     predtemp = model(input_tmp.to(device))
        #     pred.append(predtemp.cpu().detach())
        # pred = torch.concat(pred,dim=1)

        # inputs_water.append(input.cpu()[:,water_start:water_end,:])
        outputs_water.append(output.cpu()[:,water_start:water_end,:])
        preds_water.append(pred.cpu())

        metrics = cal_metrics(output.cpu()[:,water_start:water_end,:], pred.cpu().view(input.shape[0],-1,output.shape[-1]),
                              [percent_10, percent_5, percent_1])

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
    if method_name.lower() == 'nbeats'.lower():
        model = NBeats(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'NLinear'.lower():
        model = NLinear(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'TSMixer'.lower():
        model = TSMixer(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
    elif method_name.lower() == 'mlp'.lower():
        model = MLP(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
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

    # CNN
    elif method_name.lower() == 'ModernTCN'.lower():
        model = ModernTCN(input_t_length,span_t_length,output_t_length,output_dim,output_dim,output_dim)
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

            pred = model(input[:,water_start:water_end,:].to(device))
            loss = criterion(pred, output[:,water_start:water_end,:].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            count_loss += 1
            # break
        print(f'{epoch} loss:{sum_loss/count_loss}')
        metric_dict = evaluation(model,val_dataloder,dataset_dict['val'],device,batch_size)
        if metric_dict['mse']<best_eval:
            best_eval = metric_dict['mse']
            best_model_dict = model.state_dict()

    model.load_state_dict(best_model_dict)
    test_metric_dict = evaluation(model,test_dataloder,dataset_dict['test'],device,batch_size,valid=False)
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

    parser.add_argument('--dataset_path', default='../dataset_download/Processed')
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--dataset', default='S_2', choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'])
    parser.add_argument('--length_input', default='3D', choices=['1D', '2D', '3D', '1W', '2W', '3W'])
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'])
    parser.add_argument('--length_output', default='12H', choices=['1H', '6H', '12H', '1D', '2D'])

    parser.add_argument('--method', default='mlp') # S2IPLLM

    parser.add_argument('--lr', default=1E-3)
    parser.add_argument('--weight_decay', default=0E-5)
    parser.add_argument('--epoches', default=50)
    parser.add_argument('--batchsize', default=8)


    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    print(args)
    main(args)
