# -*- coding: utf-8 -*-

import h5py
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules import MLP, BiAffineAttn, TimeDistributed


class BiAffineParser(nn.Module):
    def __init__(self, n_word_vocab, n_word_embed,
                 n_pos_vocab, n_pos_embed, drop,
                 lstm_hidden, n_lstm_layers, lstm_drop,
                 arc_hidden, arc_depth, arc_drop,
                 lab_hidden, lab_depth, lab_drop,
                 n_labels):
        super(BiAffineParser, self).__init__()

        # Embeddings
        self.word_embed = TimeDistributed(nn.Embedding(n_word_vocab,
                                                       n_word_embed))
        self.pos_embed = TimeDistributed(nn.Embedding(n_pos_vocab,
                                                      n_pos_embed))
        self.drop = nn.Dropout(p=drop)

        # LSTM
        lstm_input = n_word_embed + n_pos_embed
        self.lstm = nn.LSTM(input_size=lstm_input,
                            hidden_size=lstm_hidden,
                            batch_first=True,
                            dropout=lstm_drop,
                            bidirectional=True)

        # MLPs
        self.arc_mlp_h = TimeDistributed(MLP(lstm_hidden * 2,
                                             arc_hidden,
                                             arc_depth,
                                             arc_drop))
        self.arc_mlp_d = TimeDistributed(MLP(lstm_hidden * 2,
                                             arc_hidden,
                                             arc_depth,
                                             arc_drop))

        self.lab_mlp_h = TimeDistributed(MLP(lstm_hidden * 2,
                                             lab_hidden, lab_depth,
                                             lab_drop))
        self.lab_mlp_d = TimeDistributed(MLP(lstm_hidden * 2,
                                             lab_hidden,
                                             lab_depth,
                                             lab_drop))

        # BiAffine layers
        self.arc_attn = BiAffineAttn(in_dim=arc_hidden,
                                     out_channels=1,
                                     bias_head=False,
                                     bias_dep=True)
        self.lab_attn = BiAffineAttn(in_dim=lab_hidden,
                                     out_channels=n_labels,
                                     bias_head=True,
                                     bias_dep=True)

    def load_pretrained(self, filename):
        with h5py.File(filename, 'r') as f:
            embeddings = torch.from_numpy(f['word_vectors'][:])
        self.word_embed.module.weight = nn.Parameter(embeddings)

    def forward(self, x_word, x_pos, lens):
        # Embeddings
        x_word = self.word_embed(x_word)
        x_pos = self.pos_embed(x_pos)
        x = torch.cat((x_word, x_pos), dim=-1)
        x = self.drop(x)

        # LSTM
        x = pack_padded_sequence(x, lens, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # MLPs
        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)

        # Attention
        S_arc = self.arc_attn(arc_h, arc_d)
        S_lab = self.lab_attn(lab_h, lab_d)

        return S_arc, S_lab
