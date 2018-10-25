# -*- coding: utf-8 -*-

from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, layer_size, drop):
        super(MLP, self).__init__()

        self.layer = nn.Sequential(nn.Linear(input_size, layer_size),
                                   nn.LeakyReLU(negative_slope=0.1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.layer(x)
        x = self.drop(x)

        return x
