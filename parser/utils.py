# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def init_embedding(embed):
    bias = (3. / embed.size(-1)) ** 0.5
    nn.init.uniform_(embed, -bias, bias)


def numericalize(vocab, corpus):
    x = [vocab.word_to_id(seq) for seq in corpus.word_seqs]
    char_x = [vocab.char_to_id(seq) for seq in corpus.word_seqs]
    heads = [torch.tensor(seq, dtype=torch.long) for seq in corpus.head_seqs]
    labels = [vocab.label_to_id(seq) for seq in corpus.label_seqs]
    items = [x, char_x, heads, labels]

    return items
