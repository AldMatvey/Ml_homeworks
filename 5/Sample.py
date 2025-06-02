import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import multiprocessing as mp
import os
import cv2
import torch
import torchvision

def extract_sample(n_way, n_support, n_query, datax, datay, root):
    """
    Выбор случайного сэмпла размера n_support+n_querry, for n_way classes
    Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
    Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """

    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        try:
            datax_cls = datax[datay == cls]
            perm = np.random.permutation(datax_cls)
            sample_cls = perm[:(n_support + n_query)]
            sample.append([cv2.imread(root + fname) / 255 for fname in sample_cls])
        except TypeError:
            print([fname for fname in sample_cls])
            break

    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.permute(0, 1, 4, 2, 3)
    return ({
        'images': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })

def display_sample(sample):
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))
