# -*- coding: utf-8 -*-

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, drop):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_input, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.drop = nn.Dropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight,
                            gain=nn.init.calculate_gain(self.activation))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.layer(x)
        x = self.drop(x)

        return x
