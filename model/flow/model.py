from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch
import torch.nn as nn


class RNVP(nn.Module):

    def __init__(self, cfg, means, learnable_mean=False):
        super().__init__()

        self.layer_dims = cfg['flow']['layer_dims']
        self.input_dims = cfg['flow']['input_dims']
        
        self.inn = self.build_inn(cfg)
        self.net_parameters = [p for p in self.inn.parameters() if p.requires_grad]
        for param in self.net_parameters:
            param.data = 0.05 * torch.randn_like(param)
        
        if learnable_mean:
            self.means = torch.nn.Parameter(means)
        else:
            self.register_buffer('means', means)
        

    def build_inn(self, cfg):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, self.layer_dims), nn.ReLU(),
                                nn.Linear(self.layer_dims,  self.layer_dims), nn.ReLU(),
                                nn.Linear(self.layer_dims,  c_out))
        
        nodes = [InputNode(self.input_dims, name='input')]

        for k in range(cfg['flow']['n_blocks']):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock, 
                              {'subnet_constructor': subnet_fc, 'clamp': cfg['flow']['clamping']},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes, verbose=False)


    def forward(self, x):
        return self.inn(x)

    def reverse(self, y_rev):
        return self.inn(y_rev, rev=True)
