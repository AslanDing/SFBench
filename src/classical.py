import os
import torch
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from dataloader import load_dataset_loader
from evaluation import cal_metrics

import xgboost as xgb
from sklearn.svm import SVR

import sys
from pprint import pprint
import warnings
import argparse

from tqdm import tqdm

import multiprocessing

warnings.filterwarnings('ignore')

def main_SVR(args):
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
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir, dataset,
                                                        length_input, length_span, length_output, batch_size,
                                                        cache_dir=cache_dir)

    input_t_length = dataset_dict['train'].length_input
    span_t_length = dataset_dict['train'].length_span
    output_t_length = dataset_dict['train'].length_output

    train_dataloder = dataloader_dict['train']
    val_dataloder = dataloader_dict['val']
    test_dataloder = dataloader_dict['test']


    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]


    mean = dataset_dict['train'].all_timeseries_std_mean['WATER']['mean']
    std = dataset_dict['train'].all_timeseries_std_mean['WATER']['std']

    all_train_input = []
    all_train_output = []
    for i in range(output_dim):
        all_train_input.append([])
        all_train_output.append([])

    for batch in tqdm(train_dataloder):
        # all_train_input.append(batch['WATER_input'].cpu().numpy())
        # all_train_output.append(batch['WATER_output'].cpu().numpy())
        train_input = batch['WATER_input'].cpu().numpy()
        train_output = batch['WATER_output'].cpu().numpy()
        for i in range(output_dim):
            all_train_input[i].append(train_input[:, i, :])
            all_train_output[i].append(train_output[:, i, :])

    # all_train_input = np.concatenate(all_train_input, axis=0)
    # all_train_output = np.concatenate(all_train_output, axis=0)

    all_test_input = []
    all_test_output = []
    for i in range(output_dim):
        all_test_input.append([])
        all_test_output.append([])

    for batch in tqdm(test_dataloder):
        # all_test_input.append(batch['WATER_input'].cpu().numpy())
        # all_test_output.append(batch['WATER_output'].cpu().numpy())
        train_input = batch['WATER_input'].cpu().numpy()
        train_output = batch['WATER_output'].cpu().numpy()
        for i in range(output_dim):
            all_test_input[i].append(train_input[:, i, :])
            all_test_output[i].append(train_output[:, i, :])

    all_label = all_test_output  # []
    all_forecast = []
    for i in tqdm(range(output_dim)):
        t_all_input = np.concatenate(all_train_input[i], axis=0)
        t_all_output = np.concatenate(all_train_output[i], axis=0)
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_model.fit(t_all_input, t_all_output)
        svr_pred = svr_model.predict(np.concatenate(all_test_input[i],axis=0))
        all_forecast.append(svr_pred)

    # all_label = np.stack(all_label)
    all_forecast = np.stack(all_forecast)

    percent_10 = (dataset_dict['train'].percentile_mask_10['WATER'] - mean) / std
    percent_5 = (dataset_dict['train'].percentile_mask_5['WATER'] - mean) / std
    percent_1 = (dataset_dict['train'].percentile_mask_1['WATER'] - mean) / std

    metrics = cal_metrics(all_forecast, all_label, [percent_10, percent_5, percent_1])
    pprint(metrics)

def main_XGBoost(args):
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
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir, dataset,
                                                        length_input, length_span, length_output, batch_size,
                                                        cache_dir=cache_dir)

    input_t_length = dataset_dict['train'].length_input
    span_t_length = dataset_dict['train'].length_span
    output_t_length = dataset_dict['train'].length_output

    train_dataloder = dataloader_dict['train']
    val_dataloder = dataloader_dict['val']
    test_dataloder = dataloader_dict['test']


    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]


    mean = dataset_dict['train'].all_timeseries_std_mean['WATER']['mean']
    std = dataset_dict['train'].all_timeseries_std_mean['WATER']['std']

    all_train_input = []
    all_train_output = []
    for i in range(output_dim):
        all_train_input.append([])
        all_train_output.append([])

    for batch in tqdm(train_dataloder):
        # all_train_input.append(batch['WATER_input'].cpu().numpy())
        # all_train_output.append(batch['WATER_output'].cpu().numpy())
        train_input = batch['WATER_input'].cpu().numpy()
        train_output = batch['WATER_output'].cpu().numpy()
        for i in range(output_dim):
            all_train_input[i].append(train_input[:,i,:])
            all_train_output[i].append(train_output[:,i,:])

    # all_train_input = np.concatenate(all_train_input, axis=0)
    # all_train_output = np.concatenate(all_train_output, axis=0)

    all_test_input = []
    all_test_output = []
    for i in range(output_dim):
        all_test_input.append([])
        all_test_output.append([])

    for batch in tqdm(test_dataloder):
        # all_test_input.append(batch['WATER_input'].cpu().numpy())
        # all_test_output.append(batch['WATER_output'].cpu().numpy())
        train_input = batch['WATER_input'].cpu().numpy()
        train_output = batch['WATER_output'].cpu().numpy()
        for i in range(output_dim):
            all_test_input[i].append(train_input[:,i,:])
            all_test_output[i].append(train_output[:,i,:])

    #
    # all_test_input = np.concatenate(all_test_input, axis=0)
    # all_test_output = np.concatenate(all_test_output, axis=0)

    all_label = all_test_output #[]
    all_forecast = []
    for i in tqdm(range(output_dim)):
        # n_jobs=multiprocessing.cpu_count() // 2,
        t_all_input = np.concatenate(all_train_input[i],axis=0)
        t_all_output = np.concatenate(all_train_output[i],axis=0)
        # xgb_model = xgb.XGBRegressor(reg_alpha=0.75,
        #                              reg_lambda=0.45,
        #                              subsample=0.8,
        #                              n_estimators=100, max_depth=3,
        #                              learning_rate=0.1, random_state=42,
        #                              device=device)
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(t_all_input,
                      t_all_output)

        xgb_pred = xgb_model.predict(np.concatenate(all_test_input[i],axis=0))
        all_forecast.append(xgb_pred)

    # all_label = np.stack(all_label)
    all_forecast = np.stack(all_forecast)

    percent_10 = (dataset_dict['train'].percentile_mask_10['WATER'] - mean) / std
    percent_5 = (dataset_dict['train'].percentile_mask_5['WATER'] - mean) / std
    percent_1 = (dataset_dict['train'].percentile_mask_1['WATER'] - mean) / std

    metrics = cal_metrics(all_forecast, all_label,[percent_10, percent_5, percent_1])
    pprint(metrics)

def main_ARIMA(args):
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
    dataset_dict, dataloader_dict = load_dataset_loader(data_dir, dataset,
                                                        length_input, length_span, length_output, batch_size,
                                                        cache_dir=cache_dir)
    input_dim = 0
    for key in dataset_dict['train'].all_timeseries.keys():
        dim_tmp = dataset_dict['train'].all_timeseries[key].shape[0]
        input_dim += dim_tmp
    output_dim = dataset_dict['train'].all_timeseries['WATER'].shape[0]


    mean = dataset_dict['train'].all_timeseries_std_mean['WATER']['mean']
    std = dataset_dict['train'].all_timeseries_std_mean['WATER']['std']

    all_label = []
    all_forecast = []
    for i in tqdm(range(output_dim)):
        local_mean =  mean[i]
        local_std =  std[i]
        time_series_train  = (dataset_dict['train'].all_timeseries['WATER'][i] - local_mean)/local_std
        time_series_valid  = (dataset_dict['val'].all_timeseries['WATER'][i] - local_mean)/local_std
        time_series_test = (dataset_dict['test'].all_timeseries['WATER'][i] - local_mean)/local_std

        timeseris = torch.concat([time_series_train,time_series_valid]).cpu().numpy()
        model = ARIMA(timeseris, order=(1, 1, 1)).fit()
        forecast_steps = len(time_series_test)
        forecast = model.forecast(steps=forecast_steps)

        all_label.append(time_series_test.cpu().numpy())
        all_forecast.append(forecast)

    all_label = np.stack(all_label)
    all_forecast = np.stack(all_forecast)

    percent_10 = (dataset_dict['train'].percentile_mask_10['WATER'] - mean )/std
    percent_5 = (dataset_dict['train'].percentile_mask_5['WATER'] - mean )/std
    percent_1 = (dataset_dict['train'].percentile_mask_1['WATER'] - mean )/std

    metrics = cal_metrics(all_forecast, all_label,[percent_10, percent_5, percent_1])
    pprint(metrics)

if __name__ == "__main__":
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
    parser.add_argument('--dataset', default='S_3',
                        choices=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7'])
    parser.add_argument('--length_input', default='3D', choices=['1D', '2D', '3D', '1W', '2W', '3W'])
    parser.add_argument('--length_span', default='0H', choices=['0H', '1H', '1D', '1W'])
    parser.add_argument('--length_output', default='12H', choices=['1H', '6H', '12H', '1D', '2D'])

    parser.add_argument('--method', default='AMARIA')  # S2IPLLM

    parser.add_argument('--lr', default=1E-5)
    parser.add_argument('--weight_decay', default=0E-5)
    parser.add_argument('--epoches', default=100)
    parser.add_argument('--batchsize', default=8)

    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    print(args)
    if 'AMARIA'.lower() in args.method.lower():
        main_ARIMA(args)
    elif 'XGBoost'.lower() in args.method.lower():
        main_XGBoost(args)
    elif 'SVR'.lower() in args.method.lower():
        main_SVR(args)



