# -*- coding: utf-8 -*-

import time
from parser import BiAffineParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from data_utils import batch_loader, make_dataset, split_train_test


def train(n_epochs=10):
    torch.set_num_threads(4)
    torch.manual_seef(1)
    data_file = 'data/train-stanford-raw.conll'
    # if vocab_file is given (ie for pretrained wordvectors), use x2i and i2x from this file.
    # If not given, create new vocab file in data
    vocab_file = 'data/glove.6B.100d.txt'

    print('epoch,train_loss,val_loss,arc_acc,lab_acc')

    batch_size = 64
    prints_per_epoch = 10
    n_epochs *= prints_per_epoch

    # load data
    print('loading data...')
    data, x2i, i2x = make_dataset(data_file)

    # make train and val batch loaders
    train_data, val_data = split_train_test(data)
    print('# train sentences', len(train_data))
    print('# val sentences', len(val_data))
    train_loader = batch_loader(train_data, batch_size)
    val_loader = batch_loader(val_data, batch_size, shuffle=False)

    print('creating model...')
    # make model
    model = BiAffineParser(n_word_vocab=len(x2i['word']),
                           n_word_embed=100,
                           n_pos_vocab=len(x2i['tag']),
                           n_pos_embed=28,
                           emb_drop=0.33,
                           lstm_hidden=512,
                           n_lstm_layers=4,
                           lstm_drop=0.33,
                           arc_hidden=256,
                           arc_depth=1,
                           arc_drop=0.33,
                           lab_hidden=128,
                           lab_depth=1,
                           lab_drop=0.33,
                           n_labels=len(x2i['label']))
    print(model)
    base_params, arc_params, lab_params = model.get_param_groups()

    optimizer = Adam(params=model.parameters(), lr=2e-3, betas=(0.9, 0.9))
    sched = ReduceLROnPlateau(optimizer=optimizer,
                              threshold=1e-3,
                              patience=8,
                              factor=.4,
                              verbose=True)

    n_train_batches = int(len(train_data) / batch_size)
    n_val_batches = int(len(val_data) / batch_size)
    batches_per_epoch = int(n_train_batches / prints_per_epoch)

    for epoch in range(n_epochs):
        t0 = time.time()

        # Training
        train_loss = 0
        model.train()
        for i in range(batches_per_epoch):
            optimizer.zero_grad()

            # Load batch
            words, tags, arcs, lens = next(train_loader)

            # Forward
            S_arc, S_lab = model(words, tags, lens=lens)

            # Calculate loss
            arc_loss = get_arc_loss(S_arc, arcs)
            lab_loss = get_label_loss(S_lab, arcs)
            loss = arc_loss + .025 * lab_loss
            train_loss += arc_loss.item() + lab_loss.item()

            # Backward
            loss.backward()
            optimizer.step()

        train_loss /= batches_per_epoch

        # Evaluation
        val_loss = 0
        arc_acc = 0
        lab_acc = 0
        model.eval()
        for i in range(n_val_batches):
            words, tags, arcs, lens = next(val_loader)

            S_arc, S_lab = model(words, tags, lens=lens)

            arc_loss = get_arc_loss(S_arc, arcs)
            lab_loss = get_label_loss(S_lab, arcs)
            loss = arc_loss + lab_loss

            val_loss += arc_loss.item() + lab_loss.item()
            arc_acc += get_arc_accuracy(S_arc, arcs)
            lab_acc += get_label_accuracy(S_lab, arcs)

        val_loss /= n_val_batches
        arc_acc /= n_val_batches
        lab_acc /= n_val_batches
        epoch_time = time.time() - t0

        print(f"Epoch {epoch/prints_per_epoch:.1f} "
              f"train_loss {train_loss:.3f} val_loss {val_loss:.3f} "
              f"arc_acc {arc_acc:.2%} lab_acc {lab_acc:.2%} "
              f"time {epoch_time:.1f} sec")

        print(f"{epoch / prints_per_epoch:.3f}, "
              f"{train_loss:.3f}, {val_loss:.3f}, "
              f"{arc_acc:.2%}, {lab_acc:.2%}")

        sched.step(val_loss)

    print('Done!')
    torch.save(model, 'model.pt')


def get_arc_loss(S_arc, arcs):
    """
    S_arc is a tensor of [batch, heads, deps]
    Arcs is a np array with columns [batch_idx, head, dep, label]
    Calculates softmax over columns in S_arc, S_arc[b,:,i] = P(head | dep=i)
    """
    logits = S_arc.transpose(-1, -2)[arcs[:, 0], arcs[:, 2], :]
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


def get_arc_accuracy(S_arc, arcs):
    heads = torch.from_numpy(arcs[:, 1])
    logits = S_arc.transpose(-1, -2)[arcs[:, 0], arcs[:, 2], :]
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
