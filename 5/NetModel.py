import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import multiprocessing as mp
import os
import cv2

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from metrics import euclidean_dist
from Encoder import Encoder


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    encoder = Encoder()
    return ProtoNet(encoder)

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.loss_step_values = []
        self.accuracy_step_values = []

        self.encoder = encoder.cpu()

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat (predict)
        """
        sample_images = sample['images'].cpu()
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support, x_query = (sample_images[:, :n_support].reshape(n_way * n_support, 3, 128, 128),
                              sample_images[:, n_support:].reshape(n_way * n_query, 3, 128, 128))

        x_support = self.encoder(x_support)
        x_support = x_support.reshape(n_way, n_support, x_support.shape[-1])
        x_prototype = x_support.mean(1)

        x_query = self.encoder(x_query)

        n, m = x_query.shape
        l = x_prototype.shape[0]

        dists = torch.pow(x_query.unsqueeze(1).expand(n, l, m) - x_prototype.expand(n, l, m), 2).sum(2)

        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cpu()

        log_p_y = nn.LogSoftmax(dim=-1)(-dists).reshape(n_way, n_query, n_way)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        self.loss_step_values.append(loss_val.item())
        self.accuracy_step_values.append(acc_val.item())

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
        }