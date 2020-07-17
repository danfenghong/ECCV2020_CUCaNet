#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Before run this file, please ensure running <python -m visdom.server> in current environment.
Then, please go to http:localhost://#display_port# to see the visulizations.
"""

import torch
import time
import hues
import os
from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
import scipy.io as sio
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1)

if __name__ == "__main__":

    train_opt = TrainOptions().parse()
    train_opt.name = 'CAVE_food' # 'CAVE_toy'
    train_opt.num_theta = 128
    train_opt.niter = 2000
    train_opt.niter_decay = 8000
    train_opt.lr = 5e-3
    train_opt.lr_decay_iters = 1000
    train_opt.display_port = 8097
    train_opt.data_name    = 'CAVE'
    train_opt.srf_name     = 'Nikon_D700_Qu' # 'Landsat8_BGR'
    train_opt.mat_name     = 'fake_and_real_food_ms' # 'chart_and_stuffed_toy_ms'
    train_opt.scale_factor = 32
    train_opt.print_freq   = 250
    train_opt.save_freq    = 250
    train_opt.batchsize    = 1
    train_opt.which_epoch  = train_opt.niter + train_opt.niter_decay
    train_opt.useSoftmax   = 'No'
    train_opt.isCalSP = 'Yes'
    train_opt.display_port = 8097
    
    # trade-off parameters: could be better tuned
    # for auto-reconstruction
    train_opt.lambda_A = 10
    train_opt.lambda_B = 10
    # gamma_1
    train_opt.lambda_C = 10
    # gamma_2
    train_opt.lambda_G = 10
    # alpha
    train_opt.lambda_D = 0.01
    # beta
    train_opt.lambda_E = 0.01
    # gamma_3
    train_opt.lambda_F = 10000
    
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)
    
    train_model.setup(train_opt)
    visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)
        
    total_steps = 0
    
    for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        train_psnr_list = []

        for i, data in enumerate(train_dataloader):

            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize

            visualizer.reset()

            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)

            hues.info("[{}/{} in {}/{}]".format(i,dataset_size//train_opt.batchsize,
                                                epoch,train_opt.niter + train_opt.niter_decay))

            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)

            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    visualizer.display_current_results(train_model.get_current_visuals(),
                                                       train_model.get_image_name(), epoch, True,
                                                       win_id=[1])

                    visualizer.plot_spectral_lines(train_model.get_current_visuals(), train_model.get_image_name(),
                                                   visual_corresponding_name=train_model.get_visual_corresponding_name(),
                                                   win_id=[2,3])
                    visualizer.plot_psnr_sam(train_model.get_current_visuals(), train_model.get_image_name(),
                                             epoch, float(epoch_iter) / dataset_size,
                                             train_model.get_visual_corresponding_name())

                    visualizer.plot_lr(train_model.get_LR(), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))

        train_model.update_learning_rate()
        
    rec_hhsi=train_model.get_current_visuals()[train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]
    sio.savemat(os.path.join("./Results/CAVE/",''.join(data['name'])+'.mat'),{'out': rec_hhsi.transpose(1,2,0)})
    
