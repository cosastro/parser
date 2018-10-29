# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ParserLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout_in=0, dropout_out=0,
                 bidirectional=False):
        super(ParserLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = []
        self.b_cells = []
        for i_layer in range(self.num_layers):
            layer_input_size = (input_size if i_layer ==
                                0 else hidden_size * self.num_directions)
            for i_dir in range(self.num_directions):
                cells = (self.f_cells if i_dir == 0 else self.b_cells)
                cell = nn.LSTMCell(input_size=layer_input_size,
                                   hidden_size=hidden_size)
                self.reset_parameters(cell)
                cells.append(cell)

        self.f_cells = torch.nn.ModuleList(self.f_cells)
        self.b_cells = torch.nn.ModuleList(self.b_cells)

    def reset_parameters(self, cell):
        cell.bias_ih.data.zero_()
        cell.bias_hh.data.zero_()

        nn.init.orthogonal_(cell.weight_ih)
        nn.init.orthogonal_(cell.weight_hh)

    @staticmethod
    def _forward_rnn(cell, x, mask, initial, h_zero, in_drop_masks,
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
            if self.training and self.dropout_in > 1e-3:
                in_drop_mask = torch.bernoulli(x.new_full(
                    (batch_size, x.size(2)), 1 - self.dropout_in))/(1 - self.dropout_in)

            if self.training and self.dropout_out > 1e-3:
                hid_drop_mask = torch.bernoulli(x.new_full(
                    (batch_size, self.hidden_size), 1 - self.dropout_out))/(1 - self.dropout_out)
                if self.bidirectional:
                    hid_drop_mask_b = torch.bernoulli(x.new_full(
                        (batch_size, self.hidden_size), 1 - self.dropout_out))/(1 - self.dropout_out)

            # , (layer_h_n, layer_c_n) = \
            layer_output = \
                self._forward_rnn(cell=self.f_cells[layer],
                                  x=x,
                                  mask=mask,
                                  initial=initial,
                                  h_zero=h_zero,
                                  in_drop_masks=in_drop_mask,
                                  hid_drop_masks_for_next_timestamp=hid_drop_mask,
                                  is_backward=False)

            #  only share input_dropout
            if self.bidirectional:
                b_layer_output =  \
                    self._forward_rnn(cell=self.b_cells[layer],
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
