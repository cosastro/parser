# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_utils import batch_loader, make_dataset, split_train_test
from models import BiAffineParser


def train(n_epochs=100):
    torch.set_num_threads(8)
    torch.manual_seed(1)
    ftrain = 'data/train-stanford-raw.conll'
    # if vocab_file is given (ie for pretrained wordvectors), use x2i and i2x from this file.
    # If not given, create new vocab file in data

    batch_size = 50

    print('loading data...')
    data, x2i, i2x = make_dataset(ftrain)

    train_data, dev_data = split_train_test(data)
    print('# train sentences', len(train_data))
    print('# dev sentences', len(dev_data))

    print('creating model...')
    # make model
    model = BiAffineParser(n_word_vocab=len(x2i['word']),
                           n_pos_vocab=len(x2i['tag']),
                           n_word_embed=100,
                           n_pos_embed=28,
                           n_lstm_hidden=150,
                           n_lstm_layers=2,
                           n_mlp_hidden=100,
                           n_labels=len(x2i['label']),
                           emb_drop=0.33,
                           lstm_drop=0.33,
                           mlp_drop=0.33)
    print(model)

    optimizer = optim.Adam(params=model.parameters(),
                           lr=2e-3, betas=(0.9, 0.9))
    sched = ReduceLROnPlateau(optimizer=optimizer,
                              threshold=1e-3,
                              patience=8,
                              factor=.4,
                              verbose=True)

    for epoch in range(n_epochs):
        start = datetime.now()

        # Training
        train_loss = 0
        model.train()
        for words, tags, arcs, lens in tqdm(batch_loader(train_data, batch_size),
                                            total=(len(train_data) //
                                                   batch_size),
                                            ncols=1):
            optimizer.zero_grad()

            # Forward
            s_arc, s_lab = model(words, tags, lens=lens)

            # Calculate loss
            arc_loss = get_arc_loss(s_arc, arcs)
            lab_loss = get_label_loss(s_lab, arcs)
            loss = arc_loss + lab_loss

            # Backward
            loss.backward()
            optimizer.step()
            # sched.step()

        print(f"Epoch: {epoch}:")
        loss, uas, las = evaluate(model, batch_loader(train_data, batch_size))
        print(f"{'train:':<6} Loss: {loss:.4f} UAS: {uas:.2%} LAS: {las:.2%}")
        loss, uas, las = evaluate(model, batch_loader(
            dev_data, batch_size, shuffle=False))
        print(f"{'dev:':<6} Loss: {loss:.4f} UAS: {uas:.2%} LAS: {las:.2%}")
        t = datetime.now() - start
        print(f"{t}s elapsed\n")

    print('Done!')
    torch.save(model, 'model.pt')


@torch.no_grad()
def evaluate(model, loader):
    # Evaluation
    loss, uas, las = 0, 0, 0
    model.eval()
    for words, tags, arcs, lens in tqdm(loader, ncols=1):
        s_arc, s_lab = model(words, tags, lens=lens)

        arc_loss = get_arc_loss(s_arc, arcs)
        lab_loss = get_label_loss(s_lab, arcs)
        loss += arc_loss + lab_loss

        uas += get_arc_accuracy(s_arc, arcs)
        las += get_label_accuracy(s_lab, arcs)

    loss /= len(loader)
    uas /= len(loader)
    las /= len(loader)

    return loss, uas, las


def get_arc_loss(s_arc, arcs):
    """
    s_arc is a tensor of [batch, heads, deps]
    Arcs is a np array with columns [batch_idx, head, dep, label]
    Calculates softmax over columns in s_arc, s_arc[b,:,i] = P(head | dep=i)
    """
    logits = s_arc.transpose(-1, -2)[arcs[:, 0], arcs[:, 2], :]
    heads = torch.from_numpy(arcs[:, 1])

    return F.cross_entropy(logits, heads)


def get_label_loss(S_label, arcs):
    """
    S_label is a tensor of shape [batch, n_labels, heads, deps]
    arc_labels is a list of tuples (batch_idx, head_idx, dep_idx, label)
    Calculates softmax over second dimension of S_label,
    S_label[b, :, i, j] = P(label | head=i, dep=j).
    """
    logits = S_label.permute(0, 2, 3, 1)[
        arcs[:, 0], arcs[:, 1], arcs[:, 2], :]
    labels = torch.from_numpy(arcs[:, 3])
    return F.cross_entropy(logits, labels)


def get_arc_accuracy(s_arc, arcs):
    heads = torch.from_numpy(arcs[:, 1])
    logits = s_arc.transpose(-1, -2)[arcs[:, 0], arcs[:, 2], :]
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == heads).item()
    return correct / len(arcs)


def get_label_accuracy(S_label, arcs):
    labels = torch.from_numpy(arcs[:, 3])
    logits = S_label.permute(0, 2, 3, 1)[
        arcs[:, 0], arcs[:, 1], arcs[:, 2], :]
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / len(arcs)


if __name__ == "__main__":
    train()
