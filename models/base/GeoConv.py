import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    def build(self):
        self.process_coords = nn.Linear(2, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj['lngs'], dim = 2)
        lats = torch.unsqueeze(traj['lats'], dim = 2)
        locs = torch.cat((lngs, lats), dim = 2)
        
        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords.cpu()(locs))
        locs = locs.permute(0, 2, 1)
        conv_locs = F.elu(self.conv.cpu()(locs)).permute(0, 2, 1)
        
        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'], config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim = 2)

        #Remove coordinate layer
        conv_locs = torch.cat((conv_locs, local_dist), dim = 2)
        
        return conv_locs