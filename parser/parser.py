# -*- coding: utf-8 -*-

from parser.modules import MLP, BiAffine, CharLSTM, ParserLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class BiAffineParser(nn.Module):
    def __init__(self, n_vocab, n_embed, n_char, n_char_embed, n_char_out,
                 n_lstm_hidden, n_lstm_layers, n_mlp_arc, n_mlp_lab, n_labels,
                 drop):
        super(BiAffineParser, self).__init__()

        # the embedding layer
        self.embed = nn.Embedding(n_vocab, n_embed)
        # the char-lstm layer
        self.char_lstm = CharLSTM(n_char=n_char,
                                  n_embed=n_char_embed,
                                  n_out=n_char_out)

        # the word-lstm layer
        self.lstm = ParserLSTM(input_size=n_embed + n_char_out,
                               hidden_size=n_lstm_hidden,
                               num_layers=n_lstm_layers,
                               batch_first=True,
                               dropout_in=drop,
                               dropout_out=drop,
                               bidirectional=True)

        # MLP层
        self.mlp_arc_h = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_arc,
                             drop=drop)
        self.mlp_arc_d = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_arc,
                             drop=drop)

        self.mlp_lab_h = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_lab,
                             drop=drop)
        self.mlp_lab_d = MLP(n_input=n_lstm_hidden * 2,
                             n_hidden=n_mlp_lab,
                             drop=drop)
        # BiAffine layers
        self.arc_attn = BiAffine(n_input=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.lab_attn = BiAffine(n_input=n_mlp_lab,
                                 n_out=n_labels,
                                 bias_x=True,
                                 bias_y=True)

        self.drop = nn.Dropout(p=drop)

        self.apply(self.init_weights)

    def init_weights(self, m):
        # if type(m) == nn.LSTM:
        #     for i in m.parameters():
        #         if len(i.shape) > 1:
        #             nn.init.orthogonal_(i)
        if type(m) == nn.Embedding:
            bias = (3. / m.weight.size(1)) ** 0.5
            nn.init.uniform_(m.weight, -bias, bias)

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, char_x):
        # get the mask and lengths of given batch
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        x = self.embed(x)
        char_x = self.char_lstm(char_x[mask])
        char_x = pad_sequence(torch.split(char_x, lens.tolist()), True)

        # concatenate the word and char representations
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)

        x = self.lstm(x, mask)

        # MLPs
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        lab_h = self.mlp_lab_h(x)
        lab_d = self.mlp_lab_d(x)

        # Attention
        s_arc = self.arc_attn(arc_d, arc_h)
        s_lab = self.lab_attn(lab_d, lab_h).permute(0, 2, 3, 1)

        return s_arc, s_lab

    def get_loss(self, s_arc, s_lab, heads, labels, mask):
        batch_size, maxlen = mask.size()
        loss = nn.CrossEntropyLoss(reduction='sum')
        true_mask = mask.clone()
        true_mask[:, 0] = true_mask.new_zeros(batch_size, dtype=torch.uint8)

        heads = heads[true_mask]
        labels = labels[true_mask]
        word_num = true_mask.sum()

        s_arc.masked_fill_((1-mask).unsqueeze(1), -10000)
        arc_loss = loss(s_arc[true_mask], heads)

        s_lab = s_lab[true_mask]
        s_lab = s_lab[torch.arange(word_num), heads]
        label_loss = loss(s_lab, labels)

        return (arc_loss + label_loss) / word_num.float()
