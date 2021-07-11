import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

from Model import Date2VecConvert

class Net(nn.Module):
    embed_dims = [('driverID', 24000, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]
    #d2v = Date2VecConvert(model_path="./Date2Vec/d2v_model/d2v_state_dict")

    def __init__(self):
        super(Net, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))
        #self.add_module('date_time_em', d2v)
        
    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        #sz += 64
       
        # append total distance
        return sz + 1

    def forward(self, attr):
        em_list = []
        for name, dim_in, dim_out in Net.embed_dims:
            embed = getattr(self, name + '_em')
            
            attr_t = attr[name].view(-1, 1)
            attr_t = torch.squeeze(embed(attr_t))
            em_list.append(attr_t)
    
        #d2v_t = Net.d2v(attr['date_time'])
        #em_list.append(d2v_t)

        dist = utils.normalize(attr['dist'], 'dist')
        em_list.append(dist.view(-1, 1))

        return torch.cat(em_list, dim = 1)
