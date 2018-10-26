# -*- coding: utf-8 -*-

import torch
from torch import nn


class BiAffine(nn.Module):

    def __init__(self, n_input, n_out=1, bias_x=True, bias_y=True):
        super(BiAffine, self).__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_input + bias_x,
                                                n_input + bias_y))
        self.reset_parameters()

    def reset_parameters(self):
        bias = 1. / self.weight.size(1) ** 0.5
        nn.init.uniform_(self.weight, -bias, bias)

    def forward(self, x, y):
        if self.bias_x:
            x = self.add_bias(x)
        if self.bias_y:
            y = self.add_bias(y)
        # [batch_size, 1, seq_len, d]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, d]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        s = x @ self.weight @ torch.transpose(y, -1, -2)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

    def add_bias(self, x):
        b = torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
        x = torch.cat([x, b], -1)

        return x
