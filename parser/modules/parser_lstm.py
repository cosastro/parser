# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ParserLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout=0, bidirectional=False):
        super(ParserLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                                hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def _lstm_forward(self, cell, x, mask, initial, h_zero, in_drop_masks,
                      hid_drop_masks_for_next_timestamp, is_backward):
        seq_len = x.size(0)  # length batch dim
        output = []
        # ??? What if I want to use an initial vector than can be tuned?
        hx = (initial, h_zero)
        for t in range(seq_len):
            if is_backward:
                t = seq_len - t - 1
            input_i = x[t]
            if in_drop_masks is not None:
                input_i = input_i * in_drop_masks
            h_next, c_next = cell(input=input_i, hx=hx)
            # element-wise multiply; broadcast
            h_next = h_next * mask[t] + h_zero[0] * (1 - mask[t])
            c_next = c_next * mask[t] + h_zero[1] * (1 - mask[t])
            output.append(h_next)  # NO drop for now
            if hid_drop_masks_for_next_timestamp is not None:
                h_next = h_next * hid_drop_masks_for_next_timestamp
            hx = (h_next, c_next)
        if is_backward:
            output.reverse()
        output = torch.stack(output, 0)
        return output  # , hx

    def forward(self, x, mask, initial=None):
        mask = torch.unsqueeze(mask.transpose(0, 1), dim=2).float()
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, input_size = x.shape

        h_zero = x.new_zeros((batch_size, self.hidden_size))
        if initial is None:
            initial = h_zero

        # h_n, c_n = [], []
        for layer in range(self.num_layers):
            in_drop_mask, hid_drop_mask, hid_drop_mask_b = None, None, None
            if self.training:
                in_drop_mask = torch.bernoulli(x.new_full(
                    (batch_size, x.size(2)), 1 - self.dropout))/(1 - self.dropout)
                hid_drop_mask = torch.bernoulli(x.new_full(
                    (batch_size, self.hidden_size), 1 - self.dropout))/(1 - self.dropout)
                if self.bidirectional:
                    hid_drop_mask_b = torch.bernoulli(x.new_full(
                        (batch_size, self.hidden_size), 1 - self.dropout))/(1 - self.dropout)

            # , (layer_h_n, layer_c_n) = \
            layer_output = self._lstm_forward(cell=self.f_cells[layer],
                                              x=x,
                                              mask=mask,
                                              initial=initial,
                                              h_zero=h_zero,
                                              in_drop_masks=in_drop_mask,
                                              hid_drop_masks_for_next_timestamp=hid_drop_mask,
                                              is_backward=False)

            #  only share input_dropout
            if self.bidirectional:
                b_layer_output = self._lstm_forward(cell=self.b_cells[layer],
                                                    x=x,
                                                    mask=mask,
                                                    initial=initial,
                                                    h_zero=h_zero,
                                                    in_drop_masks=in_drop_mask,
                                                    hid_drop_masks_for_next_timestamp=hid_drop_mask_b,
                                                    is_backward=True)
            #  , (b_layer_h_n, b_layer_c_n) = \
            # h_n.append(torch.cat([layer_h_n, b_layer_h_n], 1) if self.bidirectional else layer_h_n)
            # c_n.append(torch.cat([layer_c_n, b_layer_c_n], 1) if self.bidirectional else layer_c_n)
            if self.bidirectional:
                x = torch.cat([layer_output, b_layer_output], 2)
            else:
                x = layer_output

        # h_n = torch.stack(h_n, 0)
        # c_n = torch.stack(c_n, 0)
        if self.batch_first:
            x = x.transpose(0, 1)  # , (h_n, c_n)

        return x  # , (h_n, c_n)
