# -*- coding: utf-8 -*-

from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, layer_size, depth, drop):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(nn.Linear(input_size, layer_size),
                                             nn.ReLU(),
                                             nn.Dropout(drop)))
            input_size = layer_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
