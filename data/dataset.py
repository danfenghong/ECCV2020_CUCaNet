#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, args, sp_matrix, isTrain=True):
        super(Dataset, self).__init__()

        self.args = args
        self.sp_matrix = sp_matrix
        self.msi_channels = sp_matrix.shape[1]

        self.isTrain = isTrain

        default_datapath = os.getcwd()
        data_folder = os.path.join(default_datapath, args.data_name)
        if os.path.exists(data_folder):
            for root, dirs, files in os.walk(data_folder):
                if args.mat_name in files:
                    raise Exception("HSI data path does not exist!")
                else:
                    data_path = os.path.join(data_folder, args.mat_name+'.mat')
        else:
            return 0

        self.imgpath_list = sorted(glob.glob(data_path))
        self.img_list = []
        for i in range(len(self.imgpath_list)):
            self.img_list.append(io.loadmat(self.imgpath_list[i])['img'])

        'for single HSI'
        (_, _, self.hsi_channels) = self.img_list[0].shape

        'generate simulated data'
        self.img_patch_list = []
        self.img_lr_list = []
        self.img_msi_list = []
        for i, img in enumerate(self.img_list):
            (h, w, c) = img.shape
            s = self.args.scale_factor
            'Ensure that the side length can be divisible'
            r_h, r_w = h%s, w%s
            img_patch = img[int(r_h/2):h-(r_h-int(r_h/2)),int(r_w/2):w-(r_w-int(r_w/2)),:]
            self.img_patch_list.append(img_patch)
            'LrHSI'
            img_lr = self.generate_LrHSI(img_patch, s)
            self.img_lr_list.append(img_lr)
            (self.lrhsi_height, self.lrhsi_width, _) = img_lr.shape
            'HrMSI'
            img_msi = self.generate_HrMSI(img_patch, self.sp_matrix)
            self.img_msi_list.append(img_msi)

    def downsamplePSF(self, img,sigma,stride):
        def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride,stride),sigma)
        if img.ndim == 3:
            img_w,img_h,img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w,img_h = img.shape
            img = img.reshape((img_w,img_h,1))
        from scipy import signal
        out_img = np.zeros((img_w//stride, img_h//stride, img_c))
        for i in range(img_c):
            out = signal.convolve2d(img[:,:,i],h,'valid')
            out_img[:,:,i] = out[::stride,::stride]
        return out_img

    def generate_LrHSI(self, img, scale_factor):
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        return img_lr

    def generate_HrMSI(self, img, sp_matrix):
        (h, w, c) = img.shape
        self.msi_channels = sp_matrix.shape[1]
        if sp_matrix.shape[0] == c:
            img_msi = np.dot(img.reshape(w*h,c), sp_matrix).reshape(h,w,sp_matrix.shape[1])
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        return img_msi

    def __getitem__(self, index):
        img_patch = self.img_patch_list[index]
        img_lr = self.img_lr_list[index]
        img_msi = self.img_msi_list[index]
        img_name = os.path.basename(self.imgpath_list[index]).split('.')[0]
        img_tensor_lr = torch.from_numpy(img_lr.transpose(2,0,1).copy()).float()
        img_tensor_hr = torch.from_numpy(img_patch.transpose(2,0,1).copy()).float()
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2,0,1).copy()).float()
        return {"lhsi":img_tensor_lr,
                'hmsi':img_tensor_rgb,
                "hhsi":img_tensor_hr,
                "name":img_name}

    def __len__(self):
        return len(self.imgpath_list)
