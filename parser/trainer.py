# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.metrics import AttachmentMethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Trainer(object):

    def __init__(self, model, vocab, optimizer, scheduler):
        super(Trainer, self).__init__()

        self.model = model
        self.vocab = vocab
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, train_loader, dev_loader, test_loader,
            epochs, patience, file):
        total_time = timedelta()
        max_e, max_metric = 0, 0.0

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            self.train(train_loader)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, train_metric = self.evaluate(train_loader)
            print(f"{'train:':<6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {loss:.4f} {test_metric}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # save the model if it is the best so far
            if dev_metric > max_metric:
                torch.save(self.model, file)
                max_e, max_metric = epoch, dev_metric
            elif epoch - max_e >= patience:
                break
        self.model = torch.load(file).cuda()
        loss, metric = self.evaluate(test_loader)

        print(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        print(f"the score of test at epoch {max_e} is {metric.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def get_loss(self, s_arc, s_lab, heads, labels):
        arc_loss = F.cross_entropy(s_arc, heads)
        s_lab = s_lab[torch.arange(len(s_lab)), heads]
        label_loss = F.cross_entropy(s_lab, labels)
        loss = arc_loss + label_loss

        return loss

    def train(self, loader):
        self.model.train()

        for x, char_x, heads, labels in loader:
            self.optimizer.zero_grad()

            mask = x.gt(0)
            s_arc, s_lab = self.model(x, char_x)
            loss = self.get_loss(s_arc[mask], s_lab[mask],
                                 heads[mask], labels[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        loss, uas, las = 0, 0.0, 0.0
        metric = AttachmentMethod()

        for x, char_x, heads, labels in loader:
            mask = x.gt(0)
            s_arc, s_lab = self.model(x, char_x)
            s_arc = s_arc[mask]
            s_lab = s_lab[mask]
            heads = heads[mask]
            labels = labels[mask]
            loss += self.get_loss(s_arc, s_lab, heads, labels)
            pred_arcs = torch.argmax(s_arc, dim=1)
            pred_labels = torch.argmax(
                s_lab[torch.arange(len(s_lab)), heads], dim=1)
            metric(pred_arcs, pred_labels, heads, labels)
        loss /= len(loader)

        return loss, metric


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
