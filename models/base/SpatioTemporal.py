import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer

from models.base import GeoConv

from torch.autograd import Variable

class Net(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''
    def __init__(self, attr_size, kernel_size = 3, num_filter = 32, pooling_method = 'attention', rnn = 'lstm'):
        super(Net, self).__init__()
        
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.rnn_type = rnn
        
        self.geo_conv = GeoConv.Net(kernel_size = kernel_size, num_filter = num_filter)
    	
        #num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn == 'lstm':
            self.rnn = nn.GRU(input_size = num_filter + 1 + attr_size, \
                                      hidden_size = 128, \
                                      num_layers = 2, \
                                      batch_first = True
            )
                
        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size = num_filter + 1 + attr_size, \
                              hidden_size = 128, \
                              num_layers = 1, \
                              batch_first = True
            )
    
        elif rnn == 'transformer':
            # Model parameters
            d_model = 16 # Lattent dim
            q = 4 # Query size
            v = 4 # Value size
            h = 4 # Number of heads
            N = 2 # Number of encoder and decoder to stack
            attention_size = 8 # Attention window size
            dropout = 0.3 # Dropout rate
            pe = None # Positional encoding
            chunk_mode = None
            
            d_input = num_filter + 1 + attr_size # From dataset
            d_output = 128 # From dataset
            self.rnn = transformer.Transformer(d_input, 
                                   d_model, 
                                   d_output, 
                                   q, 
                                   v,
                                   h, 
                                   N, 
                                   attention_size=attention_size, 
                                   dropout=dropout, 
                                   chunk_mode=chunk_mode, 
                                   pe=pe)
        
        if pooling_method == 'attention':
            self.attr2atten = nn.Linear(attr_size, 128)
    
    def out_size(self):
        # return the output size of spatio-temporal component
        return 128
    
    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)
    
        if torch.cuda.is_available():
            lens = torch.cuda.FloatTensor(lens)
        else:
            lens = torch.FloatTensor(lens)
    
        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False)
    
        hiddens = hiddens / lens
    
        return hiddens
    
    
    def attent_pooling(self, hiddens, lens, attr_t):
        attent = F.tanh(self.attr2atten.cpu()(attr_t)).permute(0, 2, 1)
    
    	#hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)
    
        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)
    
        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)
    
        return hiddens
    
    
    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)
        
        attr_t = torch.unsqueeze(attr_t, dim = 1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))

        # concat the loc_conv and the attributes
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim = 2)
    
        lens = list(map(lambda x: x - self.kernel_size + 1, traj['lens']))
      
        if self.rnn_type == 'lstm' or self.rnn_type == 'rnn':

            packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first = True)

            packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)

            hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
            
        elif self.rnn_type == 'transformer':

            hiddens = self.rnn(conv_locs)

            packed_hiddens = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first = True)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
    
        elif self.pooling_method == 'attention':
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)