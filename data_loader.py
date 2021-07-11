import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

import pandas as pd

secties_df = pd.read_csv('C:/Users/s154193/Documents/Master Thesis/DeepTTE-master/DeepTTE-master/sectie_id.csv')
secties_dict = secties_df['Sectie'].to_dict()

def get_key(my_dict, val): 
    for key, value in my_dict.items(): 
         if value == val: 
             return key
        
def get_value(my_dict, dict_key): 
    for key, value in my_dict.items(): 
         if key == dict_key: 
             return value 

class MySet(Dataset):
    def __init__(self, input_file):
        self.content = open('./data/' + input_file, 'r').readlines()
        self.content = list(map(lambda x: json.loads(x), self.content))
        self.lengths = list(map(lambda x: len(x['lngs']), self.content))

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(list(self.content))

def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID', 'Treinnr', 'date_time']
    traj_attrs = ['lngs', 'lats', 'time_gap', 'dist_gap', 'Sectie']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        if key in ['driverID', 'dateID', 'weekID', 'timeID', 'Treinnr']:
            attr[key] = torch.LongTensor([item[key] for item in data])
        elif key in ['date_time']:
            
            attr[key] = np.asarray([item[key] for item in data])
            
            attr[key] = list(attr[key])
            for i in range(0,len(attr[key])):
                attr[key][i] = attr['date_time'][i].split('_')
                for j in range(0, len(attr[key][i])):
                    attr[key][i][j] = int(attr[key][i][j])
            
    for key in traj_attrs:
        # pad to the max length
        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            seqs = np.asarray([item[key] for item in data])
            if key in ['lngs', 'lats']:
                traj[key + '_original'] = seqs
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(seqs)

            padded = utils.normalize(padded, key)
            padded = torch.from_numpy(padded).float()
            traj[key] = padded
            
        elif key in ['Sectie']:
            seqs = np.asarray([item[key] for item in data])
            
            traj[key] = seqs
       
    lens = lens.tolist()
    traj['lens'] = lens
    
    #traj['Sectie'] = secties

    return attr, traj

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)
    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x), \
                             num_workers = 0,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader