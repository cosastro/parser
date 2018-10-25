# -*- coding: utf-8 -*-


class Config(object):
    ftrain = 'data/cdt-train-10k.conll'
    fdev = 'data/cdt-dev.conll'
    ftest = 'data/cdt-test.conll'
    fembed = 'data/giga.100.txt'
    n_embed = 100
    n_char_embed = 30
    n_char_out = 300
    n_lstm_hidden = 150
    n_lstm_layers = 2
    n_mlp_hidden = 150
