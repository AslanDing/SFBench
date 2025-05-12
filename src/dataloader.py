import os

import pandas
import torch
from sklearn.utils.fixes import percentile
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader


splits = ['S_0','S_1', 'S_2','S_3','S_4','S_5','S_6','S_7']
t_inputs = {'1D':[24,24,  24, 24, 24, 24, 24, 24],
            '2D':[48,48,  48, 48, 48, 48, 48, 48],
            '3D':[72,72,  72, 72, 72, 72, 72, 72],
            '4D':[96,96,  96, 96, 96, 96, 96, 96],
            '5D':[120,120, 120,120,120,120,120,120],
            '6D':[144,144, 144,144,144,144,144,144],
            '8D':[192,192, 192,192,192,192,192,192],
            '10D':[240,240, 240,240,240,240,240,240],
            '1W':[168,168, 168,168,168,168,168,168]}

t_spans = {'1W':[168,168, 168,168,168,168,168,168],
          '1D':[24,24,  24, 24, 24, 24, 24, 24],
          '1H':[1,1,  1, 1, 1, 1, 1, 1],
          '0H':[0,0,  0, 0, 0, 0, 0, 0]}
t_outputs = {
            '1D':[24,24,  24, 24, 24, 24, 24, 24],
            '2D':[48,48,  48, 48, 48, 48, 48, 48],
            '3D':[72,72,  72, 72, 72, 72, 72, 72],
            '5D':[120,120, 120,120,120,120,120,120],
            '1W':[168,168, 168,168,168,168,168,168],
             '12H': [12,12, 12,12,12,12,12,12],
             '6H':[6,6,  6,6,6,6,6,6],
             '1H':[1,1,  1,1,1,1,1,1]}
splits_dates = {'S_0': {'train': ['1985-01-01 00:00:00','1987-12-31 23:59:59'],
                       'val': ['1988-01-01 00:00:00', '1988-12-31 23:59:59'],
                       'test': ['1989-01-01 00:00:00','1989-12-31 23:59:59']},
                'S_1':{'train': ['1990-01-01 00:00:00','1992-12-31 23:59:59'],
                       'val': ['1993-01-01 00:00:00', '1993-12-31 23:59:59'],
                       'test': ['1994-01-01 00:00:00','1994-12-31 23:59:59']},
                'S_2': {'train': ['1995-01-01 00:00:00', '1997-12-31 23:59:59'],
                       'val': ['1998-01-01 00:00:00', '1998-12-31 23:59:59'],
                       'test': ['1999-01-01 00:00:00', '1999-12-31 23:59:59']},
                'S_3': {'train': ['2000-01-01 00:00:00', '2002-12-31 23:59:59'],
                       'val': ['2003-01-01 00:00:00', '2003-12-31 23:59:59'],
                       'test': ['2004-01-01 00:00:00', '2004-12-31 23:59:59']},
                'S_4': {'train': ['2005-01-01 00:00:00', '2007-12-31 23:59:59'],
                       'val': ['2007-01-01 00:00:00', '2008-12-31 23:59:59'],
                       'test': ['2009-01-01 00:00:00', '2009-12-31 23:59:59']},
                'S_5': {'train': ['2010-01-01 00:00:00', '2012-12-31 23:59:59'],
                       'val': ['2013-01-01 00:00:00', '2013-12-31 23:59:59'],
                       'test': ['2014-01-01 00:00:00', '2014-12-31 23:59:59']},
                'S_6': {'train': ['2015-01-01 00:00:00', '2017-12-31 23:59:59'],
                       'val': ['2018-01-01 00:00:00', '2018-12-31 23:59:59'],
                       'test': ['2019-01-01 00:00:00', '2019-12-31 23:59:59']},
                'S_7': {'train': ['2020-01-01 00:00:00', '2021-12-31 23:59:59'],
                       'val': ['2022-01-01 00:00:00', '2022-12-31 23:59:59'],
                       'test': ['2023-01-01 00:00:00', '2023-12-31 23:59:59']}}

x_str = 'X COORD'
y_str = 'Y COORD'

def sort_files(dir,split):

    all_files = {'WATER': None,
                 'RAIN': None,
                 'WELL': None,
                 'PUMP': None,
                 'GATE': None}

    for statio_type in all_files.keys():
        path_dir = f'{dir}/{statio_type}/{split}'
        dict_files = {}
        for first_level in os.listdir(path_dir):
            first_level_path = os.path.join(path_dir, first_level)
            if os.path.isdir(first_level_path):
                tmp_dict = {}
                for file in os.listdir(first_level_path):
                    if file.endswith(".json"):
                        if 'json' in tmp_dict.keys():
                            raise ValueError('two json files in one folder')
                        else:
                            tmp_dict['json']=os.path.join(first_level_path, file)
                    if file.endswith(".csv"):
                        if 'csv' in tmp_dict.keys():
                            raise ValueError('two csv files in one folder')
                        else:
                            tmp_dict['csv']=os.path.join(first_level_path, file)
                # station_name : json, csv
                dict_files[first_level] = tmp_dict

        all_files[statio_type] = dict_files

    return all_files

class SFFLOODDataset(Dataset):
    def __init__(self, dir, split, length_input,length_span,length_output,
                    training_datetime,device='cpu',ESP=1E-4,time_emb=False):
        super().__init__()
        self.time_emb = time_emb
        self.device = device
        (self.all_timeseries, self.all_timemasks, self.all_stationnames,self.all_locations,
         self.percentile_mask_10, self.percentile_mask_5,
         self.percentile_mask_1,self.datetime_emb) = self.load_csv_files(dir,split, training_datetime)
        self.length = self.all_timeseries['WATER'][0].shape[0] - length_input - length_span - length_output
        self.length_input = length_input
        self.length_span = length_span
        self.length_output = length_output
        self.ESP=ESP
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        output_dict = {}

        if self.time_emb:
            output_dict['timeenc_input'] = self.datetime_emb[:, idx:idx + self.length_input]
            output_dict['timeenc_output'] = self.datetime_emb[:,
                                            idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]

        for key in self.all_timeseries.keys():
            data = self.all_timeseries[key]
            mean = self.all_timeseries_std_mean[key]['mean']
            std = self.all_timeseries_std_mean[key]['std']
            data_mask = self.all_timemasks[key]

            output_dict[key+'_input'] = (data[:,idx:idx+self.length_input] - mean)/std
            output_dict[key+'_input_mask'] = data_mask[:,idx:idx+self.length_input]
            output_dict[key+'_output'] = (data[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output] - mean)/std
            output_dict[key+'_output_mask'] = data_mask[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output]

        return output_dict

    def set_std_mean(self,std_mean_dict):
        self.all_timeseries_std_mean = std_mean_dict

    def calculate_normalization(self,all_timeseries):

        all_timeseries_std_mean = {'WATER': None,
                          'RAIN': None,
                          'WELL': None,
                          'PUMP': None,
                          'GATE': None}

        for key in all_timeseries_std_mean.keys():
            timeseries_data = all_timeseries[key]
            tmp = timeseries_data 
            mean = torch.mean(tmp,dim=1)
            std = torch.std(tmp, dim=1)
            std = torch.where(std<self.ESP,torch.ones_like(std),std)
            all_timeseries_std_mean[key] = {'mean':mean.view(-1,1),'std':std.view(-1,1)}

        return all_timeseries_std_mean

    def time_encoding(self,datetime):

        df_stamp = pd.to_datetime(datetime)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
        data_stamp = df_stamp.drop(['TIMESTAMP'], 1).values
        return data_stamp

    def load_csv_files(self,dir,split,training_datetime):
        all_files = sort_files(dir, split)

        # load csv file
        all_timeseries = {'WATER': None,
                          'RAIN': None,
                          'WELL': None,
                          'PUMP': None,
                          'GATE': None}
        all_timemasks = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}

        percentile_mask_10 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}
        percentile_mask_5 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}
        percentile_mask_1 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}

        all_stationnames = {'WATER': None,
                            'RAIN': None,
                            'WELL': None,
                            'PUMP': None,
                            'GATE': None}

        all_station_loc = {'WATER': None,
                            'RAIN': None,
                            'WELL': None,
                            'PUMP': None,
                            'GATE': None}

        datetime = None

        for key in all_timeseries.keys():
            print('Load time series : ', key)
            station_name_list = []
            timeseries_list = []
            timemasks_list = []
            percentile_mask_10_list = []
            percentile_mask_5_list = []
            percentile_mask_1_list = []
            gate_Latitude = []
            gate_Longitude = []
            files_dict = all_files[key]
            for station_name in tqdm(files_dict.keys()):
                csv_file_path = files_dict[station_name]['csv']
                data_frame = pd.read_csv(csv_file_path)

                if data_frame['INTERPOLATED_VALUE'].isnull().any():
                    print('station_name:',station_name)
                    continue

                json_file = files_dict[station_name]['json']
                with open(json_file) as fp:
                    dd = json.load(fp)
                    gate_Latitude.append(float(dd[x_str]))
                    gate_Longitude.append(float(dd[y_str]))


                timeseries_data = data_frame['INTERPOLATED_VALUE'].values
                lower_threshold = np.percentile(timeseries_data, 5)
                upper_threshold = np.percentile(timeseries_data, 95)
                percentile_mask_10_list.append(np.array([lower_threshold,upper_threshold]))

                lower_threshold = np.percentile(timeseries_data, 2.5)
                upper_threshold = np.percentile(timeseries_data, 97.5)
                percentile_mask_5_list.append(np.array([lower_threshold,upper_threshold]))

                lower_threshold = np.percentile(timeseries_data, 0.5)
                upper_threshold = np.percentile(timeseries_data, 99.5)
                percentile_mask_1_list.append(np.array([lower_threshold,upper_threshold]))

                data_frame['TIMESTAMP_'] = pd.to_datetime(data_frame['TIMESTAMP'])
                mask = (data_frame['TIMESTAMP_'] >= pd.to_datetime(training_datetime[0])) & (
                        data_frame['TIMESTAMP_'] <= pd.to_datetime(training_datetime[1]))
                new_df = data_frame[mask]

                timeseries_data = new_df['INTERPOLATED_VALUE'].values
                timeseries_mask = new_df['CONFIDENCE'].values
                timeseries_mask = np.where(timeseries_mask > 0, np.ones_like(timeseries_mask),
                                           np.zeros_like(timeseries_mask))

                timeseries_list.append(timeseries_data)
                timemasks_list.append(timeseries_mask)
                station_name_list.append(station_name)

            percentile_mask_10_list = np.stack(percentile_mask_10_list)
            percentile_mask_5_list = np.stack(percentile_mask_5_list)
            percentile_mask_1_list = np.stack(percentile_mask_1_list)
            timeseries_list = np.stack(timeseries_list)
            timemasks_list = np.stack(timemasks_list)

            percentile_mask_10[key] = torch.from_numpy(percentile_mask_10_list).float().to(self.device)
            percentile_mask_5[key] = torch.from_numpy(percentile_mask_5_list).float().to(self.device)
            percentile_mask_1[key] = torch.from_numpy(percentile_mask_1_list).float().to(self.device)
            all_timeseries[key] = torch.from_numpy(timeseries_list).float().to(self.device)
            all_timemasks[key] = torch.from_numpy(timemasks_list).float().to(self.device)
            all_stationnames[key] = station_name_list
            all_station_loc[key] = np.array([gate_Latitude,gate_Longitude])

        datetime_embedding = self.time_encoding(datetime)
        return (all_timeseries, all_timemasks, all_stationnames,all_station_loc,
                percentile_mask_10, percentile_mask_5, percentile_mask_1,datetime_embedding)

class SFFLOODDatasetP(Dataset):
    def __init__(self, dir, split,part, length_input,length_span,length_output,
                    training_datetime,device='cpu',ESP=1E-4,time_emb=False):
        super().__init__()
        self.time_emb = time_emb
        self.device = device
        (self.all_timeseries, self.all_timemasks, self.all_stationnames,self.all_locations,
         self.percentile_mask_10, self.percentile_mask_5,
         self.percentile_mask_1,self.datetime_emb) = self.load_csv_files(dir,split,part, training_datetime)
        self.length = self.all_timeseries['WATER'][0].shape[0] - length_input - length_span - length_output
        self.length_input = length_input
        self.length_span = length_span
        self.length_output = length_output
        self.ESP = ESP

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        output_dict = {}
        
        if self.time_emb:
            output_dict['timeenc_input'] = self.datetime_emb[:, idx:idx + self.length_input]
            output_dict['timeenc_output'] = self.datetime_emb[:,
                                            idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]

        for key in self.all_timeseries.keys():
            data = self.all_timeseries[key]
            if data is None:
                continue
            mean = self.all_timeseries_std_mean[key]['mean']
            std = self.all_timeseries_std_mean[key]['std']
            data_mask = self.all_timemasks[key]
            output_dict[key+'_input'] = (data[:,idx:idx+self.length_input] - mean)/std
            output_dict[key+'_input_mask'] = data_mask[:,idx:idx+self.length_input]
            output_dict[key+'_output'] = (data[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output] - mean)/std
            output_dict[key+'_output_mask'] = data_mask[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output]

        return output_dict

    def set_std_mean(self,std_mean_dict):
        self.all_timeseries_std_mean = std_mean_dict

    def time_encoding(self,datetime):

        df_stamp = pd.to_datetime(datetime)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
        data_stamp = df_stamp.drop(['TIMESTAMP'], 1).values
        return data_stamp


    def calculate_normalization(self,all_timeseries):

        all_timeseries_std_mean = {'WATER': None,
                          'RAIN': None,
                          'WELL': None,
                          'PUMP': None,
                          'GATE': None}

        for key in all_timeseries_std_mean.keys():
            if all_timeseries[key] is None:
                continue
            tmp = all_timeseries[key]
            mean = torch.mean(tmp,dim=1)
            std = torch.std(tmp, dim=1)
            std = torch.where(std<self.ESP,torch.ones_like(std),std)
            all_timeseries_std_mean[key] = {'mean':mean.view(-1,1),'std':std.view(-1,1)}

        return all_timeseries_std_mean

    def load_csv_files(self,dir,split,part,training_datetime):
        all_files = sort_files(dir, split)
        idx = splits.index(split)
        part = dir + f'/threeparts_{part}_map_locations_{idx}.json'
        with open(part, 'r') as f:
            files_fliter_dict = json.load(f)

        # load csv file
        all_timeseries = {'WATER': None,
                          'RAIN': None,
                          'WELL': None,
                          'PUMP': None,
                          'GATE': None}
        all_timemasks = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}

        percentile_mask_10 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}
        percentile_mask_5 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}
        percentile_mask_1 = {'WATER': None,
                         'RAIN': None,
                         'WELL': None,
                         'PUMP': None,
                         'GATE': None}

        all_stationnames = {'WATER': None,
                            'RAIN': None,
                            'WELL': None,
                            'PUMP': None,
                            'GATE': None}

        all_station_loc = {'WATER': None,
                            'RAIN': None,
                            'WELL': None,
                            'PUMP': None,
                            'GATE': None}

        datetime = None

        for key in all_timeseries.keys():
            print('Load time series : ', key)
            station_name_list = []
            timeseries_list = []
            timemasks_list = []
            percentile_mask_10_list = []
            percentile_mask_5_list = []
            percentile_mask_1_list = []
            gate_Latitude = []
            gate_Longitude = []
            files_dict = all_files[key]
            filter_name = files_fliter_dict[key]
            if len(filter_name)<1:
                continue
            for station_name in tqdm(files_dict.keys()):
                if station_name not in filter_name:
                    continue
                csv_file_path = files_dict[station_name]['csv']
                data_frame = pd.read_csv(csv_file_path)

                if data_frame['INTERPOLATED_VALUE'].isnull().any():
                    print('station_name:',station_name)
                    continue

                if datetime is None:
                    data_frame = data_frame['TIMESTAMP']

                json_file = files_dict[station_name]['json']
                with open(json_file) as fp:
                    dd = json.load(fp)
                    gate_Latitude.append(float(dd[x_str]))
                    gate_Longitude.append(float(dd[y_str]))


                timeseries_data = data_frame['INTERPOLATED_VALUE'].values
                lower_threshold = np.percentile(timeseries_data, 5)
                upper_threshold = np.percentile(timeseries_data, 95)
                percentile_mask_10_list.append(np.array([lower_threshold,upper_threshold]))

                lower_threshold = np.percentile(timeseries_data, 2.5)
                upper_threshold = np.percentile(timeseries_data, 97.5)
                percentile_mask_5_list.append(np.array([lower_threshold,upper_threshold]))

                lower_threshold = np.percentile(timeseries_data, 0.5)
                upper_threshold = np.percentile(timeseries_data, 99.5)
                percentile_mask_1_list.append(np.array([lower_threshold,upper_threshold]))

                data_frame['TIMESTAMP_'] = pd.to_datetime(data_frame['TIMESTAMP'])
                mask = (data_frame['TIMESTAMP_'] >= pd.to_datetime(training_datetime[0])) & (
                        data_frame['TIMESTAMP_'] <= pd.to_datetime(training_datetime[1]))
                new_df = data_frame[mask]

                timeseries_data = new_df['INTERPOLATED_VALUE'].values
                timeseries_mask = new_df['CONFIDENCE'].values
                timeseries_mask = np.where(timeseries_mask > 0, np.ones_like(timeseries_mask),
                                           np.zeros_like(timeseries_mask))

                timeseries_list.append(timeseries_data)
                timemasks_list.append(timeseries_mask)
                station_name_list.append(station_name)

            percentile_mask_10_list = np.stack(percentile_mask_10_list)
            percentile_mask_5_list = np.stack(percentile_mask_5_list)
            percentile_mask_1_list = np.stack(percentile_mask_1_list)
            timeseries_list = np.stack(timeseries_list)
            timemasks_list = np.stack(timemasks_list)

            percentile_mask_10[key] = torch.from_numpy(percentile_mask_10_list).float().to(self.device)
            percentile_mask_5[key] = torch.from_numpy(percentile_mask_5_list).float().to(self.device)
            percentile_mask_1[key] = torch.from_numpy(percentile_mask_1_list).float().to(self.device)
            all_timeseries[key] = torch.from_numpy(timeseries_list).float().to(self.device)
            all_timemasks[key] = torch.from_numpy(timemasks_list).float().to(self.device)
            all_station_loc[key] = np.array([gate_Latitude,gate_Longitude])
            all_stationnames[key] = station_name_list

        datetime_embedding = self.time_encoding(datetime)
        return (all_timeseries, all_timemasks, all_stationnames,all_station_loc,
                percentile_mask_10, percentile_mask_5, percentile_mask_1,datetime_embedding)

def load_dataset_loader(dir='../dataset_download/Processed_hour', split = 'S_0',
                        t_input = '3D', t_span = '0H', t_output = '12H',
                        batch_size = 128, shuffle = True, cache_dir='./cache',device='cpu',
                        use_cache = False, time_emb= False):
    """
    :param dir: Path to dataset folder
    :param spilt:  S_0-S_7
    :param t_input:  Time Series input length
    :return: dataloader, dataset
    """
    assert split in splits
    assert t_input in t_inputs

    cache_path = cache_dir + '/' +f'{split}_{t_input}_{t_span}_{t_output}.pt'
    if use_cache: #os.path.exists(cache_path):
        dataset_dict = torch.load(cache_path, weights_only=False)
        train_dataset = dataset_dict['train']
        test_dataset = dataset_dict['test']
        valid_dataset = dataset_dict['valid']
    else:

        index = splits.index(split)
        splits_date = splits_dates[split]

        length_input = t_inputs[t_input][index]
        length_span = t_spans[t_span][index]
        length_output = t_outputs[t_output][index]

        train_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['train'],device=device,time_emb=time_emb)
        all_timeseries_mean_std = train_dataset.calculate_normalization(train_dataset.all_timeseries)
        train_dataset.set_std_mean(all_timeseries_mean_std)

        valid_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['val'],device=device,time_emb=time_emb)
        valid_dataset.set_std_mean(all_timeseries_mean_std)

        test_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['test'],device=device,time_emb=time_emb)
        test_dataset.set_std_mean(all_timeseries_mean_std)

        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'train':train_dataset,'valid':valid_dataset,'test':test_dataset},cache_path)

    train_loader = DataLoader(train_dataset,batch_size,shuffle)
    valid_loader = DataLoader(valid_dataset,batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)


    return ({'train':train_dataset, 'val': valid_dataset, 'test':test_dataset},
            {'train':train_loader,'val':valid_loader,'test':test_loader})


def load_dataset_part_loader(dir='../dataset_download/Processed_hour', split = 'S_0',
                        t_input = '3D', t_span = '0H', t_output = '1D',
                        batch_size = 128, shuffle = True, cache_dir='./cache',device = 'cpu',
                        use_cache = False):

    assert split in splits
    assert t_input in t_inputs

    cache_path = cache_dir + '/' +f'{split}_{t_input}_{t_span}_{t_output}_parts.pth'
    if use_cache: # os.path.exists(cache_path):
        dataset_dict = torch.load(cache_path,weights_only=False)
        train_dataset = dataset_dict['train']
        test_dataset = dataset_dict['test']
        valid_dataset = dataset_dict['valid']
    else:

        index = splits.index(split)
        splits_date = splits_dates[split]

        length_input = t_inputs[t_input][index]
        length_span = t_spans[t_span][index]
        length_output = t_outputs[t_output][index]

        train_dataset = []
        test_dataset = []
        valid_dataset = []
        for i in range(3):
            train_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['train'],device=device,time_emb=time_emb)
            all_timeseries_mean_std = train_dataset_tmp.calculate_normalization(train_dataset_tmp.all_timeseries)
            train_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            train_dataset.append(train_dataset_tmp)

            valid_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['val'],device=device,time_emb=time_emb)
            valid_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            valid_dataset.append(valid_dataset_tmp)

            test_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['test'],device=device,time_emb=time_emb)
            test_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            test_dataset.append(test_dataset_tmp)
            
        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'train':train_dataset,'valid':valid_dataset,'test':test_dataset},cache_path)

    train_loader = []
    valid_loader = []
    test_loader = []
    for i in range(3):
        train_loader.append(DataLoader(train_dataset[i],batch_size,shuffle))
        valid_loader.append(DataLoader(valid_dataset[i],batch_size))
        test_loader.append(DataLoader(test_dataset[i],batch_size=batch_size))

    return ({'train':train_dataset, 'val': valid_dataset, 'test':test_dataset},
            {'train':train_loader,'val':valid_loader,'test':test_loader})


if __name__=="__main__":
    dataset_dict,dataloder_dict = load_dataset_loader(dir='../dataset/Processed_hour', split = 'S_0')
    data_valid = dataset_dict['val']
    length = len(data_valid)
    for idx in range(length):
        data = data_valid[idx]
        data_water_input = data['WATER_input'].numpy()
        data_water_output = data['WATER_output'].numpy()

