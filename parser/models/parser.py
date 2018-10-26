# -*- coding: utf-8 -*-

from parser.modules import MLP, BiAffine, CharLSTM

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class BiAffineParser(nn.Module):
    def __init__(self, n_vocab, n_embed, n_char, n_char_embed, n_char_out,
                 n_lstm_hidden, n_lstm_layers, n_mlp_hidden, n_labels, drop):
        super(BiAffineParser, self).__init__()

        # the embedding layer
        self.embed = nn.Embedding(n_vocab, n_embed)
        # the char-lstm layer
        self.char_lstm = CharLSTM(n_char=n_char,
                                  n_embed=n_char_embed,
                                  n_out=n_char_out)

        # the word-lstm layer
        self.word_lstm = nn.LSTM(input_size=n_embed + n_char_out,
                                 hidden_size=n_lstm_hidden,
                                 num_layers=n_lstm_layers,
                                 batch_first=True,
                                 dropout=drop,
                                 bidirectional=True)

        # MLP层
        self.arc_mlp_h = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_hidden,
                             drop=drop)
        self.arc_mlp_d = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_hidden,
                             drop=drop)

        self.lab_mlp_h = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_hidden,
                             drop=drop)
        self.lab_mlp_d = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_hidden,
                             drop=drop)
        # BiAffine layers
        self.arc_attn = BiAffine(n_input=n_mlp_hidden,
                                 bias_head=True,
                                 bias_dep=False)
        self.lab_attn = BiAffine(n_input=n_mlp_hidden,
                                 n_out=n_labels,
                                 bias_head=True,
                                 bias_dep=True)

        self.drop = nn.Dropout(p=drop)

        self.reset_parameters()

    def reset_parameters(self, m):
        bias = (3. / self.embed.weight.size(1)) ** 0.5
        nn.init.uniform_(self.embed.weight, -bias, bias)

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, char_x):
        # get the mask and lengths of given batch
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        x = self.embed(x[mask])
        char_x = self.char_lstm(char_x[mask])

        # concatenate the word and char representations
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)

        x = pack_sequence(torch.split(x, lens.tolist()))
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, True)

        # MLPs
        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)

        # Attention
        s_arc = self.arc_attn(arc_d, arc_h).permute(0, 2, 1)
        s_lab = self.lab_attn(lab_d, lab_h).permute(0, 3, 2, 1)

        return s_arc, s_lab
