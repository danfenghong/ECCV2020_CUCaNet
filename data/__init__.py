#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import numpy as np
import os
import torch
from .dataset import Dataset

def get_spectral_response(data_name, srf_name):
    xls_path = os.path.join(os.getcwd(), data_name, srf_name + '.xls')
    if not os.path.exists(xls_path):
        raise Exception("Spectral response path does not exist!")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    num_cols_sta = 1
    cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(num_cols_sta,num_cols)]
    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))
    return sp_data

def create_dataset(arg, sp_matrix, isTRain):
    dataset_instance = Dataset(arg, sp_matrix, isTRain)
    return dataset_instance

def get_sp_range(sp_matrix):
    HSI_bands, MSI_bands = sp_matrix.shape
    assert(HSI_bands>MSI_bands)
    sp_range = np.zeros([MSI_bands,2])
    for i in range(0,MSI_bands):
        index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
        sp_range[i,0] = index_dim_0[0]
        sp_range[i,1] = index_dim_0[-1]
    return sp_range

class DatasetDataLoader():
    def init(self, arg, isTrain=True):
        self.sp_matrix = get_spectral_response(arg.data_name, arg.srf_name)
        self.sp_range = get_sp_range(self.sp_matrix)
        self.dataset = create_dataset(arg, self.sp_matrix, isTrain)
        self.hsi_channels = self.dataset.hsi_channels
        self.msi_channels = self.dataset.msi_channels
        self.lrhsi_height = self.dataset.lrhsi_height
        self.lrhsi_width  = self.dataset.lrhsi_width
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=arg.batchsize if isTrain else 1,
                                                      shuffle=arg.isTrain if isTrain else False,
                                                      num_workers=arg.nThreads if arg.isTrain else 0)
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def get_dataloader(arg, isTrain=True):
    instant_dataloader = DatasetDataLoader()
    instant_dataloader.init(arg, isTrain)
    return instant_dataloader
