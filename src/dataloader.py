import os

import pandas
import torch
from sklearn.utils.fixes import percentile
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader


splits = ['S_0','S_1', 'S_2','S_3','S_4','S_5','S_6','S_7']
t_inputs = {'1D':[48,48,  144, 144, 144, 144, 144, 144],
            '2D':[96,96,  288, 288, 288, 288, 288, 288],
            '3D':[144,144,  432, 432, 432, 432, 432, 432],
            '1W':[336,336, 1008,1008,1008,1008,1008,1008],  # 1 week
            '2W':[672,672, 2016,2016,2016,2016,2016,2016],
            '3W':[1008,1008, 3024,3024,3024,3024,3024,3024],
            '4W':[1344,1344, 4032,4032,4032,4032,4032,4032]}
t_spans = {'1W':[336,336, 1008,1008,1008,1008,1008,1008],
          '1D':[48,48,  144, 144, 144, 144, 144, 144],
          '1H':[2,2,  6, 6, 6, 6, 6, 6],
          '0H':[0,0,  0, 0, 0, 0, 0, 0]}
t_outputs = {'2D':[96,96,  288, 288, 288, 288, 288, 288],
            '1D':[48,48,  144, 144, 144, 144, 144, 144],
             '12H': [24, 24, 72, 72, 72, 72, 72, 72],
             '6H':[12,12,  36, 36, 36, 36, 36, 36],
             '1H':[2,2,  6, 6, 6, 6, 6, 6]}
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
                    training_datetime,device='cpu'):
        super().__init__()
        self.device = device
        (self.all_timeseries, self.all_timemasks, self.all_stationnames,
         self.percentile_mask_10, self.percentile_mask_5,
         self.percentile_mask_1) = self.load_csv_files(dir,split, training_datetime)
        self.length = self.all_timeseries['WATER'][0].shape[0] - length_input - length_span - length_output
        self.length_input = length_input
        self.length_span = length_span
        self.length_output = length_output

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        output_dict = {}
        for key in self.all_timeseries.keys():
            data = self.all_timeseries[key]
            mean = self.all_timeseries_std_mean[key]['mean']
            std = self.all_timeseries_std_mean[key]['std']
            data_mask = self.all_timemasks[key]
            # data_mask_per10 = self.percentile_mask_10[key]
            # data_mask_per5 = self.percentile_mask_5[key]
            # data_mask_per1 = self.percentile_mask_1[key]
            output_dict[key+'_input'] = (data[:,idx:idx+self.length_input] - mean)/std
            output_dict[key+'_input_mask'] = data_mask[:,idx:idx+self.length_input]
            output_dict[key+'_output'] = (data[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output] - mean)/std
            output_dict[key+'_output_mask'] = data_mask[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output]

            # output_dict[key + '_output_per10'] = data_mask_per10[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]
            # output_dict[key + '_output_per5'] = data_mask_per5[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]
            # output_dict[key + '_output_per1'] = data_mask_per1[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]

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
            mean = torch.mean(timeseries_data,dim=1)
            std = torch.std(timeseries_data, dim=1)+1E-8
            all_timeseries_std_mean[key] = {'mean':mean.view(-1,1),'std':std.view(-1,1)}

        return all_timeseries_std_mean

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
        for key in all_timeseries.keys():
            print('Load time series : ', key)
            station_name_list = []
            timeseries_list = []
            timemasks_list = []
            percentile_mask_10_list = []
            percentile_mask_5_list = []
            percentile_mask_1_list = []
            files_dict = all_files[key]
            for station_name in tqdm(files_dict.keys()):
                csv_file_path = files_dict[station_name]['csv']
                data_frame = pd.read_csv(csv_file_path)

                if data_frame['INTERPOLATED_VALUE'].isnull().any():
                    print('station_name:',station_name)
                    continue

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
        return (all_timeseries, all_timemasks, all_stationnames,
                percentile_mask_10, percentile_mask_5, percentile_mask_1)

class SFFLOODDatasetP(Dataset):
    def __init__(self, dir, split,part, length_input,length_span,length_output,
                    training_datetime,device='cpu'):
        super().__init__()
        self.device = device
        (self.all_timeseries, self.all_timemasks, self.all_stationnames,
         self.percentile_mask_10, self.percentile_mask_5,
         self.percentile_mask_1) = self.load_csv_files(dir,split,part, training_datetime)
        self.length = self.all_timeseries['WATER'][0].shape[0] - length_input - length_span - length_output
        self.length_input = length_input
        self.length_span = length_span
        self.length_output = length_output

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        output_dict = {}
        for key in self.all_timeseries.keys():
            data = self.all_timeseries[key]
            if data is None:
                continue
            mean = self.all_timeseries_std_mean[key]['mean']
            std = self.all_timeseries_std_mean[key]['std']
            data_mask = self.all_timemasks[key]
            # data_mask_per10 = self.percentile_mask_10[key]
            # data_mask_per5 = self.percentile_mask_5[key]
            # data_mask_per1 = self.percentile_mask_1[key]
            output_dict[key+'_input'] = (data[:,idx:idx+self.length_input] - mean)/std
            output_dict[key+'_input_mask'] = data_mask[:,idx:idx+self.length_input]
            output_dict[key+'_output'] = (data[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output] - mean)/std
            output_dict[key+'_output_mask'] = data_mask[:,idx+self.length_input+self.length_span:idx+self.length_input+self.length_span+self.length_output]

            # output_dict[key + '_output_per10'] = data_mask_per10[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]
            # output_dict[key + '_output_per5'] = data_mask_per5[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]
            # output_dict[key + '_output_per1'] = data_mask_per1[:,
            #                                     idx + self.length_input + self.length_span:idx + self.length_input + self.length_span + self.length_output]

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
            if all_timeseries[key] is None:
                continue
            timeseries_data = all_timeseries[key]
            mean = torch.mean(timeseries_data,dim=1)
            std = torch.std(timeseries_data, dim=1)+1E-8
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
        for key in all_timeseries.keys():
            print('Load time series : ', key)
            station_name_list = []
            timeseries_list = []
            timemasks_list = []
            percentile_mask_10_list = []
            percentile_mask_5_list = []
            percentile_mask_1_list = []
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
        return (all_timeseries, all_timemasks, all_stationnames,
                percentile_mask_10, percentile_mask_5, percentile_mask_1)


def load_dataset_loader(dir='../dataset_download/Processed', split = 'S_0',
                        t_input = '2D', t_span = '0H', t_output = '1D',
                        batch_size = 128, shuffle = True):
    """
    :param dir: Path to dataset folder
    :param spilt:  S_0-S_7
    :param t_input:  Time Series input length
    :return: dataloader, dataset
    """
    assert split in splits
    assert t_input in t_inputs

    index = splits.index(split)
    splits_date = splits_dates[split]

    length_input = t_inputs[t_input][index]
    length_span = t_spans[t_span][index]
    length_output = t_outputs[t_output][index]

    train_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['train'])
    all_timeseries_mean_std = train_dataset.calculate_normalization(train_dataset.all_timeseries)
    train_dataset.set_std_mean(all_timeseries_mean_std)
    train_loader = DataLoader(train_dataset,batch_size,shuffle)

    valid_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['val'])
    valid_dataset.set_std_mean(all_timeseries_mean_std)
    valid_loader = DataLoader(valid_dataset,batch_size)

    test_dataset = SFFLOODDataset(dir,split, length_input,length_span,length_output, splits_date['test'])
    test_dataset.set_std_mean(all_timeseries_mean_std)
    test_loader = DataLoader(test_dataset,batch_size)

    return ({'train':train_dataset, 'val': valid_dataset, 'test':test_dataset},
            {'train':train_loader,'val':valid_loader,'test':test_loader})


def load_dataset_part_loader(dir='../dataset_download/Processed', split = 'S_0',
                        t_input = '3D', t_span = '0H', t_output = '1D',
                        batch_size = 128, shuffle = True, cache_dir='./cache'):

    assert split in splits
    assert t_input in t_inputs

    cache_path = cache_dir + '/' +f'{split}_{t_input}_{t_span}_{t_output}_parts.pth'
    if os.path.exists(cache_path):
        dataset_dict = torch.load(cache_path,map_location='cpu',weights_only=False)
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
            train_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['train'])
            all_timeseries_mean_std = train_dataset_tmp.calculate_normalization(train_dataset_tmp.all_timeseries)
            train_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            train_dataset.append(train_dataset_tmp)

            valid_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['val'])
            valid_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            valid_dataset.append(valid_dataset_tmp)

            test_dataset_tmp = SFFLOODDatasetP(dir, split, i,length_input, length_span, length_output, splits_date['test'])
            test_dataset_tmp.set_std_mean(all_timeseries_mean_std)
            test_dataset.append(test_dataset_tmp)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        torch.save({'train':train_dataset,'valid':valid_dataset,'test':test_dataset},cache_path)

    train_loader = []
    valid_loader = []
    test_loader = []
    for i in range(3):
        train_loader.append(DataLoader(train_dataset[i],batch_size,shuffle))
        valid_loader.append(DataLoader(valid_dataset[i],batch_size))
        test_loader.append(DataLoader(test_dataset[i],batch_size))

    return ({'train':train_dataset, 'val': valid_dataset, 'test':test_dataset},
            {'train':train_loader,'val':valid_loader,'test':test_loader})


if __name__=="__main__":
    dataset_dict,dataloder_dict = load_dataset_part_loader(dir='../dataset_download/Processed', split = 'S_0')
    dataset = dataset_dict['train'][0]
    data = dataset[0]
    dataset = dataset_dict['train'][2]
    data = dataset[0]

    print('')
