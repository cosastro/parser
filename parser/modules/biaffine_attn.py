# -*- coding: utf-8 -*-

import torch
from torch import nn


class BiAffineAttn(nn.Module):
    """
    BiAffine Attention layer from https://arxiv.org/abs/1611.01734
    Expects inputs as batch-first sequences [batch_size, seq_length, dim].

    Returns score matrices as [batch_size, dim, dim] for arc attention
    (n_channels=1), and score as [batch_size, n_channels, dim, dim]
    for label attention (where n_channels=#labels).
    """

    def __init__(self, n_input, n_channels, bias_head=True, bias_dep=True):
        super(BiAffineAttn, self).__init__()
        self.bias_head = bias_head
        self.bias_dep = bias_dep
        self.U = nn.Parameter(torch.Tensor(n_channels,
                                           n_input + int(bias_head),
                                           n_input + int(bias_dep)))
        self.reset_parameters()

    def reset_parameters(self):
        bias = 1. / self.U.size(1) ** 0.5
        nn.init.uniform_(self.U, -bias, bias)

    def forward(self, Rh, Rd):
        """
        Returns S = (Rh @ U @ Rd.T) with dims [batch_size, n_channels, t, t]
        S[b, c, i, j] = Score sample b Label c Head i Dep j
        """

        if self.bias_head:
            Rh = self.add_ones_col(Rh)
        if self.bias_dep:
            Rd = self.add_ones_col(Rd)

        # Add dimension to Rh and Rd for batch matrix products,
        # shape [batch, t, d] -> [batch, 1, t, d]
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)

        S = Rh @ self.U @ torch.transpose(Rd, -1, -2)

        # If n_channels == 1, squeeze [batch, 1, t, t] -> [batch, t, t]
        return S.squeeze(1)

    @staticmethod
    def add_ones_col(X):
        """
        Add column of ones to each matrix in batch.
        """
        b = torch.ones(X.shape[:-1], device=X.device).unsqueeze(-1)

        return torch.cat([X, b], -1)
