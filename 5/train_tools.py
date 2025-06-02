import torch.optim as optim
import datetime
from tqdm.notebook import tnrange
from tqdm import trange
import numpy as np
import torch

from Sample import extract_sample, display_sample


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, root):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """
    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = 'logs/tensorboard/train/' + current_time

    globaliter = 0
    log_interval = 200

    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0
    stop = False
    min_loss = np.inf
    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in trange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y, root)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
            loss.backward()
            optimizer.step()

            globaliter += 1
            if output['loss'] < min_loss:
                min_loss = output['loss']
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


        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
        epoch += 1
        scheduler.step()

    return model.accuracy_step_values, model.loss_step_values