# -*- coding: utf-8 -*-

import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AttachmentMethod(Metric):

    def __init__(self, eps=1e-5):
        super(AttachmentMethod, self).__init__()

        self.correct_arcs = 0.0
        self.correct_labels = 0.0
        self.total = 0.0
        self.eps = eps

    def __call__(self, pred_arcs, pred_labels, gold_arcs, gold_labels):
        arc_mask = (pred_arcs == gold_arcs)
        label_mask = (pred_labels == gold_labels)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_labels += (arc_mask * label_mask).sum().item()
        self.total += len(pred_arcs)

    def __repr__(self):
        return f"LAS: {self.las:.2%} UAS: {self.uas:.2%}"

    @property
    def score(self):
        return self.las

    @property
    def las(self):
        return self.correct_labels / (self.total + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)
