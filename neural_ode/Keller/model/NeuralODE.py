import numpy as np
import torch
import torch.nn as nn


class NeuralODE(nn.Module):
    '''The neural ODE'''

    def __init__(self, num_in_features, num_out_features):
        super(NeuralODE, self).__init__()
        self.act    = nn.GELU()
        
        self.l_in   = nn.Linear(
            in_features = num_in_features+1,
            out_features= 80
        )
        self.l1   = nn.Linear(
            in_features = 80,
            out_features= 80
        )
        self.l2   = nn.Linear(
            in_features = 80,
            out_features= 80
        )
        self.l_out   = nn.Linear(
            in_features = 80,
            out_features= num_out_features
        )

    def forward(self, t, y):
        '''The forward pass of the neural network, the ODE function.
        This is a tricky one. For a given t, many y could be multi-dimensional 
        based on batch size. So t needs to be turned into an array.
        '''
        
        #create shape of time array
        shape = list(y.shape)
        shape[-1] = 1
        shape = tuple(shape)
        
        #time array
        tt = torch.ones(shape) * torch.remainder(t, 1)
        #concatenate with y
        x = torch.cat((tt, y), dim=len(y.shape)-1)
        
        x = self.act(self.l_in(x))
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l_out(x)