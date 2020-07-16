import argparse
import model
import os
import torch
from utils import util

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # data
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument("--data_name", type=str, default='cave')
        parser.add_argument("--scale_factor", type=int, default=8, help='4,8')
        # gpu
        parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # class
        parser.add_argument("--model_name", type=str, default='cu_nets', help='cu_nets')
        # save
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # visualizer
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8899, help='visdom port of the web display')

        # simulation low resolution hsi method
        parser.add_argument('--sigma', type=float, default=0.5)

        self.initialize = True
        return parser

    def parse(self):
        opt = self.gater_options()
        opt.isTrain = self.isTrain

#        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


    def gater_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        parser.parse_args(['--epoch_count','2'])
        
        opt, _ = parser.parse_known_args()

        model_name = opt.model_name
        model_options_setter = model.get_option_setter(model_name)
        parser = model_options_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()
        # assert(opt.data_name == opt.dataset_name.split('_')[1])

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
