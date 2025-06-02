import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathlib import Path
import os
import cv2
import scipy
from scipy import ndimage

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as nnf

import facenet_pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1

import time

from DataLoader import create_dataset
from Sample import extract_sample, display_sample
from NetModel import *

from train_tools import train
from test_tools import test, final_test, test_batch

from tqdm.notebook import tnrange
from tqdm import trange

import pandas as pd

import datetime
import tensorflow

import plotly.express as px

from PIL import Image

import sklearn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN



if __name__ == '__main__':
    print(torch.__version__)
    print(torchvision.__version__)
    # print(facenet_pytorch.__version__)
    print(np.__version__)
    print(pd.__version__)
    print(cv2.__version__)
    print(scipy.__version__)
    print(sklearn.__version__)
    print(torch.cuda.is_available())

    # exit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))


    # Preprocess images
    # ---------------------------------------------------------------------------------------------------------
    data_root = './data/'
    images_root = data_root + 'img_align_celeba/'
    processed_images_root = data_root + 'data_mtcnn/'
    img_list = os.listdir(images_root)
    fp_identity = data_root + 'identity_CelebA.txt'

    print(f'images num: {len(img_list)}')

    mtcnn = MTCNN(image_size=128)

    df = pd.read_csv(fp_identity, sep='\s+', header=None)
    # df.to_csv('my_file.csv', header=None)
    print(f'unique persons num: {len(df[1].unique())}')
    t_start = time.time()
    count_mtcnn_not_detected = 0
    for fname in img_list:
        img = Image.open(images_root + fname)
        img = mtcnn(img)
        if img != None:
            img = torch.Tensor(cv2.normalize(img.numpy(), None, 0, 1.0,
                                             cv2.NORM_MINMAX, dtype=cv2.CV_32F)).permute(1, 2, 0)
            cv2.imwrite(processed_images_root + fname, img.numpy() * 255)
        else:
            count_mtcnn_not_detected += 1
            img = torch.Tensor(
                cv2.resize(cv2.cvtColor(cv2.imread(images_root + fname), cv2.COLOR_BGR2RGB),
                           (128, 128)))
            img = cv2.normalize(img.numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(processed_images_root + fname, img * 255)
    print(f"mtcnn couldn't find {count_mtcnn_not_detected} faces")
    t_end = time.time()
    t = t_end - t_start
    print(f"preprocessing images exec_time: {t} sec")
    plt.imshow(cv2.imread(processed_images_root + "000199.jpg"))
# ---------------------------------------------------------------------------------------------------------
    #
    trainx, trainy = create_dataset((15, 30), df)
    testx, testy = create_dataset((31, 35), df)
    print(trainx.shape, testx.shape)

    sample_example = extract_sample(5, 3, 3, trainx, trainy, processed_images_root)
    print(sample_example['images'][0].mean())
    display_sample(sample_example['images'])
#    print("device_now_is:", device)
    model = load_protonet_conv().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_way = 20
    n_support = 5
    n_query = 5
    #
    train_x = trainx
    train_y = trainy
    #
    max_epoch = 6  # 5
    epoch_size = 400 #50

    # model:
    print(f'model:\n{list(model.parameters())}')

    plt.show()

    # Training
    # # ---------------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()
    t_start = time.time()
    acc, loss = train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size,
                      processed_images_root)
    t_end = time.time()

    t = t_end - t_start
    print(f"train_exec_time: {t} sec")
    x_steps = [i for i in range(len(acc))]

    plt.show()
    plt.title('Facenet learinig')
    plt.plot(x_steps[10:], acc[10:], label='accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title('Facenet learinig')
    plt.plot(x_steps[10:], loss[10:], label='loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------------------------------------
    # Saving model
    # ---------------------------------------------------------------------------------------------------------
    model_savepath = "./model_states/"
    file1 = "model_state.txt"
    file2 = "model.pt"
    try:
        torch.save(model.state_dict(), model_savepath + file1)
    except:
        print("Error saving model.state_dict()")
    try:
        torch.save(model, model_savepath + file2)
    except:
        print("Error saving model itself")
    try:
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save('model_scripted.pt')  # Save
    except:
        print("Error saving model scripted")
    # ---------------------------------------------------------------------------------------------------------
    # Example: loading model
    model_loaded = load_protonet_conv(x_dim=(3, 128, 128),
                                      hid_dim=128,
                                      z_dim=128, )
    model_loaded.load_state_dict(torch.load(model_savepath + file1))
    model_loaded.eval().to(device)


    # Testing
    # ---------------------------------------------------------------------------------------------------------
    n_way = 10
    n_support = 1
    n_query = 5

    test_x = testx
    test_y = testy

    test_episode = 500

    test(model_loaded, test_x, test_y, n_way, n_support, n_query, test_episode, processed_images_root)
    # ---------------------------------------------------------------------------------------------------------

    my_sample = extract_sample(n_way, n_support, n_query, test_x, test_y, processed_images_root)
    display_sample(my_sample['images'])
    my_loss, my_output = model_loaded.set_forward_loss(my_sample)
    print(my_output)

    # Final test
    # ---------------------------------------------------------------------------------------------------------
    images = []
    images_dir = "./sigma_images/"
    for fname in os.listdir(images_dir):
        # img = Image.open(images_dir + fname)
        # if img == None:
        #     cv2.imread(images_dir + fname)
        # images.append(img)
        images.append(images_dir + fname)

    test_images = []
    test_dir = "./test_images/"
    for fname in os.listdir(test_dir):
        # img = Image.open(test_dir + fname)
        # if img == None:
        #     cv2.imread(test_dir + fname)
        # test_images.append(img)
        images.append(test_dir + fname)

    rest_images = []
    for fname in np.random.choice(np.unique(testx), 10, replace=False):
        rest_images.append(cv2.imread(processed_images_root + fname) / 255)
    final_test(test_images, images, rest_images, mtcnn, model_loaded, device)

    # ---------------------------------------------------------------------------------------------------------
