import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    
    return net

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        
        return loss

def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    
    return s1 + s2

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss

class ResBlock(nn.Module):
    def __init__(self, input_ch):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        )
    def forward(self, x):
        out = self.net(x)
        return out + x

def define_mut_1(input_ch, kernel_sz, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = my_mut_1(input_c=input_ch, kernel_s=kernel_sz, ngf=64)
    return init_net(net, init_type, init_gain, gpu_ids)

class my_mut_1(nn.Module):
    def __init__(self, input_c, kernel_s, ngf=64):
        super(my_mut_1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*4, kernel_s, 1, int(kernel_s/2)),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.net(x)
    
def my_define_msi2s_1(input_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = my_Msi2Delta_1(input_c=input_ch, ngf=64)
    return init_net(net, init_type, init_gain, gpu_ids)

class my_Msi2Delta_1(nn.Module):
    def __init__(self, input_c, ngf=64):
        super(my_Msi2Delta_1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*2, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2, ngf*4, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.net(x)
    
def my_define_msi2s_2(output_ch, gpu_ids, init_type='kaiming', init_gain=0.02, useSoftmax='Yes'):
    net = my_Msi2Delta_2(output_c=output_ch, ngf=64, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)

class my_Msi2Delta_2(nn.Module):
    def __init__(self, output_c, ngf=64, useSoftmax='Yes'):
        super(my_Msi2Delta_2, self).__init__()
        self.net = nn.Sequential(
           nn.Conv2d(ngf*16, output_c, 1, 1, 0)
        )
        self.usesoftmax = useSoftmax
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        if self.usesoftmax == "Yes":
            return self.softmax(self.net(x))
        elif self.usesoftmax == 'No':
            return self.net(x).clamp_(0,1)
        
def define_s2img(input_ch, output_ch,gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)

class S2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(S2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
        )
        
    def forward(self, x):
        return self.net(x).clamp_(0,1)
        
def my_define_lr2s_1(input_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = my_Lr2Delta_1(input_c=input_ch, ngf=64)
    return init_net(net, init_type, init_gain, gpu_ids)

class my_Lr2Delta_1(nn.Module):
    def __init__(self, input_c, ngf=64):
        super(my_Lr2Delta_1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf*2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4, ngf*8, 1, 1, 0),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.net(x)

def my_define_lr2s_2(output_ch, gpu_ids, init_type='kaiming', init_gain=0.02, useSoftmax='Yes'):
    net = my_Lr2Delta_2(output_c=output_ch, ngf=64, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)

class my_Lr2Delta_2(nn.Module):
    def __init__(self, output_c, ngf=64, useSoftmax='Yes'):
        super(my_Lr2Delta_2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ngf*16, output_c, 1, 1, 0)
        )
        self.usesoftmax = useSoftmax
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        if self.usesoftmax == "Yes":
            return self.softmax(self.net(x))
        elif self.usesoftmax == 'No':
            return self.net(x).clamp_(0,1)
        
def define_spectral_AM(input_ch, input_hei, input_wid, gpu_ids, init_type='mean_channel', init_gain=0.02):
    net = spectral_AM(input_c=input_ch, output_c=input_ch, input_h=input_hei, input_w=input_wid)
    return init_net(net, init_type, init_gain, gpu_ids)

class spectral_AM(nn.Module):
    def __init__(self, input_c, output_c, input_h, input_w):
        super(spectral_AM, self).__init__()
        self.net = nn.Conv2d(input_c, output_c, (input_h, input_w), 1, 0, groups=input_c)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.softmax(self.net(x))

def define_spatial_AM(input_ch, kernel_sz, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = spatial_AM(input_c=input_ch, kernel_s=kernel_sz)
    return init_net(net, init_type, init_gain, gpu_ids)

class spatial_AM(nn.Module):
    def __init__(self, input_c, kernel_s):
        super(spatial_AM, self).__init__()
        self.net = nn.Conv2d(input_c, 1, kernel_s, 1, padding=int((kernel_s-1)/2))
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, height, width = x.size()
        SAmap = self.net(x).view(b, -1, height*width)
        return self.softmax(SAmap).view(b, 1, height, width)

def define_psf(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)

class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1) # same as groups=input_c, i.e. channelwise conv
    
def define_psf_2(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF_2(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)

class PSF_2(nn.Module):
    def __init__(self, scale):
        super(PSF_2, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)

def define_hr2msi(args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type='mean_channel', init_gain=0.02):
    if args.isCalSP == "No":
        net = matrix_dot_hr2msi(sp_matrix)
    elif args.isCalSP == "Yes":
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)

class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:,1] - self.sp_range[:,0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()

        self.conv2d_list = nn.ModuleList([nn.Conv2d(x,1,1,1,0,bias=False) for x in self.length_of_each_band])
    
    def forward(self, input):
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:,self.sp_range[i,0]:self.sp_range[i,1]+1,:,:]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list,1).clamp_(0,1)

class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer('sp_matrix', torch.tensor(spectral_response_matrix.transpose(1,0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(self.sp_matrix.expand(batch,-1,-1),
                         torch.reshape(x, (batch, channel_hsi, heigth*width))).view(batch,channel_msi_sp, heigth, width)
        return hmsi

class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1e8)

class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1)

class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0,10)
                w.div_(w.sum(dim=1,keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0,5)
