#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import itertools
from . import network
from .base_model import BaseModel
import numpy as np

class CUNets(BaseModel):
    def name(self):
        return 'CUNets'

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):

        parser.set_defaults(no_dropout=True)
        if isTrain:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for lr_lr')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for msi_msi    beta')
            parser.add_argument('--lambda_C', type=float, default=1.0, help='weight for msi_s_lr   alpha')
            parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for sum2one    mu')
            parser.add_argument('--lambda_E', type=float, default=1.0, help='weight for sparse     nu')
            parser.add_argument('--lambda_F', type=float, default=1.0, help='weight for lrmsi      gamma')
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for msi_s_msi  ?')
            parser.add_argument('--lambda_H', type=float, default=0.0, help='non')
            parser.add_argument('--num_theta', type=int, default=128)
            parser.add_argument('--n_res', type=int, default=3)
            parser.add_argument('--avg_crite', type=str, default='No')
            parser.add_argument('--isCalSP', type=str, default='No')
            parser.add_argument("--useSoftmax", type=str, default='Yes')
        return parser


    def initialize(self, opt, hsi_channels, msi_channels, lrhsi_hei, lrhsi_wid, sp_matrix, sp_range):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.visual_names = ['real_lhsi', 'rec_lr_lr']
        num_s = self.opt.num_theta
        ngf = 64
        # net getnerator
        self.net_G_LR2s_1 = network.my_define_lr2s_1(input_ch=hsi_channels, gpu_ids=self.gpu_ids)
        self.net_G_LR2s_2 = network.my_define_lr2s_2(output_ch=num_s, gpu_ids=self.gpu_ids, useSoftmax=opt.useSoftmax)
        self.net_G_s2img = network.define_s2img(input_ch=num_s, output_ch=hsi_channels, gpu_ids=self.gpu_ids)
        
        self.net_G_MSI2S_1 = network.my_define_msi2s_1(input_ch=msi_channels, gpu_ids=self.gpu_ids)
        self.net_G_MSI2S_2 = network.my_define_msi2s_2(output_ch=num_s, gpu_ids=self.gpu_ids, useSoftmax=opt.useSoftmax)
        self.net_G_ms2img = network.define_s2img(input_ch=num_s, output_ch=msi_channels, gpu_ids=self.gpu_ids)
        
        self.net_G_PSF = network.define_psf(scale=opt.scale_factor, gpu_ids=self.gpu_ids)        
        self.net_G_PSF_2 = network.define_psf_2(scale=opt.scale_factor,gpu_ids=self.gpu_ids)
        self.net_G_HR2MSI = network.define_hr2msi(args=self.opt, hsi_channels=hsi_channels, msi_channels=msi_channels, sp_matrix=sp_matrix, sp_range=sp_range, gpu_ids=self.gpu_ids)
        
        self.net_G_mut_spa = network.define_spatial_AM(input_ch=ngf*8, kernel_sz=3, gpu_ids=self.gpu_ids)
        self.net_G_mut_spe = network.define_spectral_AM(input_ch=ngf*8, input_hei=int(lrhsi_hei), input_wid=int(lrhsi_wid), gpu_ids=self.gpu_ids)
        # LOSS
        if self.opt.avg_crite == "No":
            self.criterionL1Loss = torch.nn.L1Loss(size_average=False).to(self.device)
        else:
            self.criterionL1Loss = torch.nn.L1Loss(size_average=True).to(self.device)
            
        self.criterionPixelwise = self.criterionL1Loss
        self.criterionSumToOne = network.SumToOneLoss().to(self.device)
        self.criterionSparse = network.SparseKLloss().to(self.device)
        
        self.model_names = ['G_MSI2S_1', 'G_MSI2S_2', 'G_ms2img', 'G_LR2s_1', 'G_LR2s_2', 'G_s2img', 
                            'G_PSF', 'G_HR2MSI', 'G_PSF_2', 'G_mut_spa', 'G_mut_spe']
        self.setup_optimizers()
        self.visual_corresponding_name = {}

    def setup_optimizers(self, lr=None):
        if lr == None:
            lr = self.opt.lr
        else:
            isinstance(lr, float)
            lr = lr
        self.optimizers = []
        # 0.5
        self.optimizer_G_MSI2S_1 = torch.optim.Adam(itertools.chain(self.net_G_MSI2S_1.parameters()), lr=lr*0.5,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_MSI2S_1)
        self.optimizer_G_MSI2S_2 = torch.optim.Adam(itertools.chain(self.net_G_MSI2S_2.parameters()), lr=lr*0.5,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_MSI2S_2)
        self.optimizer_G_ms2img = torch.optim.Adam(itertools.chain(self.net_G_ms2img.parameters()), lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_ms2img)
        
        self.optimizer_G_LR2s_1 = torch.optim.Adam(itertools.chain(self.net_G_LR2s_1.parameters()), lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_LR2s_1)
        self.optimizer_G_LR2s_2 = torch.optim.Adam(itertools.chain(self.net_G_LR2s_2.parameters()), lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_LR2s_2)  
        self.optimizer_G_s2img = torch.optim.Adam(itertools.chain(self.net_G_s2img.parameters()), lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_s2img)
        # 0.2
        self.optimizer_G_PSF = torch.optim.Adam(itertools.chain(self.net_G_PSF.parameters()), lr=lr*0.2,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_PSF)
        self.optimizer_G_PSF_2 = torch.optim.Adam(itertools.chain(self.net_G_PSF_2.parameters()), lr=lr*0.2,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_PSF_2)

        self.optimizer_G_mut_spa = torch.optim.Adam(itertools.chain(self.net_G_mut_spa.parameters()), lr=lr*0.2,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_mut_spa)
        self.optimizer_G_mut_spe = torch.optim.Adam(itertools.chain(self.net_G_mut_spe.parameters()), lr=lr*0.2,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_G_mut_spe)
            
        if self.opt.isCalSP == 'Yes':
            # 0.2
            self.optimizer_G_HR2MSI = torch.optim.Adam(itertools.chain(self.net_G_HR2MSI.parameters()), lr=lr*0.2,betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G_HR2MSI)

    def set_input(self, input, isTrain=True):
        if isTrain:
            self.real_lhsi = Variable(input['lhsi'], requires_grad=True).to(self.device)
            self.real_hmsi = Variable(input['hmsi'], requires_grad=True).to(self.device)
            self.real_hhsi = Variable(input['hhsi'], requires_grad=True).to(self.device)

        else:
            with torch.no_grad():
                self.real_lhsi = Variable(input['lhsi'], requires_grad=False).to(self.device)
                self.real_hmsi = Variable(input['hmsi'], requires_grad=False).to(self.device)
                self.real_hhsi = Variable(input['hhsi'], requires_grad=False).to(self.device)

        self.image_name = input['name']
        self.real_input = input

    def my_forward(self):
        # LrHSI, HrMSI to themselves
        self.my_rec_lr_s_1 = self.net_G_LR2s_1(self.real_lhsi)
        self.my_rec_msi_s_1 = self.net_G_MSI2S_1(self.real_hmsi)
     
        self.my_rec_lr_s_2 = self.net_G_LR2s_2(torch.cat([self.my_rec_lr_s_1, torch.mul(self.my_rec_lr_s_1, self.net_G_PSF_2(self.net_G_mut_spa(self.my_rec_msi_s_1)))], dim=1))
        self.my_rec_msi_s_2 = self.net_G_MSI2S_2(torch.cat([self.my_rec_msi_s_1, torch.mul(self.my_rec_msi_s_1, self.net_G_mut_spe(self.my_rec_lr_s_1))], dim=1))
        
        self.rec_lr_lr = self.net_G_s2img(self.my_rec_lr_s_2)
        self.rec_msi_msi = self.net_G_ms2img(self.my_rec_msi_s_2)
        
        # HrHSI to LrHSI, HrMSI
        self.rec_msi_hr = self.net_G_s2img(self.my_rec_msi_s_2)
        self.rec_msi_lrs_lr = self.net_G_PSF(self.rec_msi_hr)
        self.rec_msi_lrs_msi = self.net_G_HR2MSI(self.rec_msi_hr)
        
        # LrHSI, HrMSI to LrMSI
        self.rec_lrhsi_lrmsi = self.net_G_HR2MSI(self.real_lhsi)
        self.rec_hrmsi_lrmsi = self.net_G_PSF(self.real_hmsi)

        self.visual_corresponding_name['real_lhsi'] = 'rec_lr_lr'
        self.visual_corresponding_name['real_hmsi'] = 'rec_msi_msi'
        self.visual_corresponding_name['real_hhsi'] = 'rec_msi_hr'
        
    def my_backward_g_joint(self, epoch):
        # lr-1
        self.loss_lr_pixelwise = self.criterionPixelwise(self.real_lhsi, self.rec_lr_lr) * self.opt.lambda_A
        self.loss_lr_s_sumtoone = self.criterionSumToOne(self.my_rec_lr_s_2) * self.opt.lambda_D
        self.loss_lr_sparse = self.criterionSparse(self.my_rec_lr_s_2) * self.opt.lambda_E
        self.loss_lr = self.loss_lr_pixelwise + self.loss_lr_s_sumtoone + self.loss_lr_sparse
        # lr-2: PSF
        self.loss_msi_ss_lr =  self.criterionPixelwise(self.real_lhsi, self.rec_msi_lrs_lr) * self.opt.lambda_G
        # msi-1
        self.loss_msi_pixelwise = self.criterionPixelwise(self.real_hmsi, self.rec_msi_msi) * self.opt.lambda_B
        self.loss_msi_s_sumtoone = self.criterionSumToOne(self.my_rec_msi_s_2) * self.opt.lambda_D
        self.loss_msi_sparse = self.criterionSparse(self.my_rec_msi_s_2) * self.opt.lambda_E
        self.loss_msi = self.loss_msi_pixelwise + self.loss_msi_s_sumtoone + self.loss_msi_sparse
        # msi-2: SRF
        self.loss_msi_ss_msi =  self.criterionPixelwise(self.real_hmsi, self.rec_msi_lrs_msi) * self.opt.lambda_C
        # lrmsi
        self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_F

        
        self.loss_joint = self.loss_lr  + self.loss_msi  + self.loss_msi_ss_lr + self.loss_msi_ss_msi + self.loss_lrmsi_pixelwise
        self.loss_joint.backward(retain_graph=False)

    def optimize_joint_parameters(self, epoch):
        self.loss_names = ["lr_pixelwise", 'lr_s_sumtoone', 'lr_sparse', 'lr',
                           'msi_pixelwise','msi_s_sumtoone','msi_sparse','msi',
                           'msi_ss_lr', 'lrmsi_pixelwise']
        self.visual_names = ['real_lhsi', 'rec_lr_lr', 'real_hmsi', 'rec_msi_msi', 'real_hhsi', 'rec_msi_hr']
            
        self.set_requires_grad([self.net_G_LR2s_1, self.net_G_LR2s_2, self.net_G_s2img, self.net_G_ms2img, 
                                self.net_G_MSI2S_1, self.net_G_MSI2S_2, self.net_G_PSF, self.net_G_HR2MSI,
                                self.net_G_PSF_2,  self.net_G_mut_spa, self.net_G_mut_spe], True)
        
        self.my_forward()
        
        self.optimizer_G_LR2s_1.zero_grad()
        self.optimizer_G_LR2s_2.zero_grad()
        self.optimizer_G_s2img.zero_grad()

        self.optimizer_G_MSI2S_1.zero_grad()
        self.optimizer_G_MSI2S_2.zero_grad()
        self.optimizer_G_ms2img.zero_grad()

        self.optimizer_G_PSF.zero_grad()
        self.optimizer_G_PSF_2.zero_grad()
                
        self.optimizer_G_mut_spa.zero_grad()        
        self.optimizer_G_mut_spe.zero_grad()

        if self.opt.isCalSP == 'Yes':
            self.optimizer_G_HR2MSI.zero_grad()
        
        self.my_backward_g_joint(epoch)

        self.optimizer_G_LR2s_1.step()
        self.optimizer_G_LR2s_2.step()
        self.optimizer_G_ms2img.step()
        
        self.optimizer_G_MSI2S_1.step()
        self.optimizer_G_MSI2S_2.step()
        self.optimizer_G_s2img.step()
        
        self.optimizer_G_PSF.step()
        self.optimizer_G_PSF_2.step()
        
        self.optimizer_G_mut_spa.step()        
        self.optimizer_G_mut_spe.step()  
        
        if self.opt.isCalSP == 'Yes':
            self.optimizer_G_HR2MSI.step()

        cliper_zeroone = network.ZeroOneClipper()
        self.net_G_s2img.apply(cliper_zeroone)
        self.net_G_ms2img.apply(cliper_zeroone)

        if self.opt.isCalSP == 'Yes':
            cliper_sumtoone = network.SumToOneClipper()
            self.net_G_HR2MSI.apply(cliper_sumtoone)

    def get_visual_corresponding_name(self):
        return self.visual_corresponding_name

    def cal_psnr(self):
        real_hsi = self.real_hhsi.data.cpu().float().numpy()[0]
        rec_hsi = self.rec_msi_hr.data.cpu().float().numpy()[0]
        return self.compute_psnr(real_hsi, rec_hsi)

    def compute_psnr(self, img1, img2):
        assert img1.ndim == 3 and img2.ndim ==3

        img_c, img_w, img_h = img1.shape
        ref = img1.reshape(img_c, -1)
        tar = img2.reshape(img_c, -1)
        msr = np.mean((ref - tar)**2, 1)
        max2 = np.max(ref)**2
        psnrall = 10*np.log10(max2/msr)
        out_mean = np.mean(psnrall)
        return out_mean
    
    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]['lr'] * 2 * 1000
        return lr
