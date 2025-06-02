from Sample import extract_sample, display_sample
from tqdm.notebook import tnrange
from tqdm import trange
import os
import cv2
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode, root):
    running_loss = 0.0
    running_acc = 0.0
    for episode in trange(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y, root)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))


def test_batch(mtcnn, my_test_path, batch_list = None): # first is support
    if batch_list == None:
        batch_list = []
    for fname in os.listdir(my_test_path):
        img = Image.open(my_test_path+"/"+fname)
        img = mtcnn(img)
        if img != None:
            img = torch.Tensor(cv2.normalize(img.numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)).permute(1,2,0)
        else:
            img = torch.Tensor(cv2.resize(cv2.cvtColor(cv2.imread("my_img/"+fname), cv2.COLOR_BGR2RGB), (128, 128)))
            img = cv2.normalize(img.numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        batch_list.append(img.numpy())
    return batch_list



def final_test(test_samples_filenames, my_samples_filenames, rest_images, mtcnn, model, device):
    batch_list = []
    samples_filenames = test_samples_filenames + my_samples_filenames
    print(f"Final test for {len(samples_filenames)} samples")
    for img_filename in samples_filenames:
        img = Image.open(img_filename)
        img = mtcnn(img)
        if img != None:
            img = torch.Tensor(cv2.normalize(img.numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)).permute(1, 2, 0)
        else:
            img = torch.Tensor(cv2.resize(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB), (128, 128)))
            img = cv2.normalize(img.numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        batch_list.append(img)

    batch_list += rest_images

    batch_list = np.array(batch_list)
    batch_list = torch.from_numpy(batch_list).float()

    sample_4D = batch_list.view(batch_list.shape[0], *batch_list.shape[1:])
    sample_4D = sample_4D.permute(0, 3, 1, 2)
    out = torchvision.utils.make_grid(sample_4D, nrow=6)
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))

    # my_support = batch_list[0].permute(2, 0, 1).unsqueeze(0).cuda()
    # my_query = batch_list[1:].permute(0, 3, 1, 2).cuda()
    my_support = batch_list[0].permute(2, 0, 1).unsqueeze(0).to(device)
    my_query = batch_list[1:].permute(0, 3, 1, 2).to(device)


    print(my_support.shape)

    my_support_emb = model.encoder(my_support)
    my_query_emb = model.encoder(my_query)
    X = np.concatenate((my_support_emb.cpu().detach().numpy(), my_query_emb.cpu().detach().numpy()))

    clustering = DBSCAN(eps=0.6, min_samples=1)  # .fit(X)
    clustering.core_sample_indices_ = np.array([0])
    clustering.fit(X)
    labels = clustering.labels_
    eps = 0.001
    batch_list[batch_list < eps] = 0
    print(batch_list.min())

    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(len(batch_list)):

        ax = fig.add_subplot(5, 13, i + 1, xticks=[], yticks=[])
        ax.imshow(sample_4D[i].permute(1, 2, 0))

        if labels[i] == labels[0]:
            ax.text(0, 0, "SIGMA", color='red')
        else:
            ax.text(0, 0, "SMBD", color='black')

    plt.show()

    return batch_list



def write_test_results(fileres, model, test_x, test_y, n_way, n_support, n_query, test_episode):
    running_loss = 0.0
    running_acc = 0.0
    samples = []
    outputs = []
    accs = []
    for episode in range(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        samples.append(sample)
        loss, output = model.set_forward_loss(sample)
        outputs.append(output)
        running_loss += output['loss']
        running_acc += output['acc']
        accs.append(output['acc'])
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results:')
    print(f'loss: {avg_loss}, accuracy: {avg_acc}')

    with open(fileres, "w") as file:
        file.write(f"Test results for {len(samples)} samples:\n")
        file.write(f"loss: {avg_loss}, accuracy: {avg_acc}\n")
        for i in range(len(samples)):
            file.write(f"sample {i} type&len: {type(samples[i])}, {len(samples[i])}\n")
            file.write(f"output {i}: {outputs[i]}\n")
            file.write(f"acc: {accs[i]}\n")

        file.write(f"\n" * 30)
        file.write(f"Samples:\n")
        for i in range(len(samples)):
            file.write(f"sample: {samples[i]}\n")