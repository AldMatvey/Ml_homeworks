#1. преобразование изображений в векторы. Делается через Image2Vector
#2. кластеризация - через k средних (Думаю, можно использовать DBSCAN также)
#3. приходит запрос, вычисляем его расстояние (евклидово) между всеми кластерами. Берем softmax (он возвращает вероятности)
#4. берем класс с более высокой вероятностью
#5. используем log-softmax в качестве loss-function
#6. Датасет omniglot
#7. Сначала нужно реализовать image2vector через 4 простых шага.
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.nn import TripletMarginLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import joblib
import torch.nn as nn
import torch
from pathlib import Path
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def symbol_name(path):
    c_slash = 0
    i = 0
    name = ""
    dest = 'train'
    if int(path[-6:-4]) > 17:
        dest = 'test'
    while True:
        if path[i] == '\\':
            c_slash += 1
            i += 1
            if c_slash == 4:
                name += " "
            continue
        if c_slash == 3 or c_slash == 4:
            name += path[i]
        elif c_slash > 4:
            break
        i+= 1
    return name, dest

#===============Подготовим датасет==========================

dataset = dict()
testset = dict()

pathlist = Path("./data").glob('**/*.png')
for path in pathlist:
    symbol, dest = symbol_name(str(path))
    if dest == 'train':
        if not symbol in dataset:
            dataset[symbol] = list()
        dataset[symbol].append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    else:
        if not symbol in testset:
            testset[symbol] = list()
        testset[symbol].append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

#===============Реализация encoder'a==========================

K = 6492 # кол-во классов с учетом поворота
N = 20 # кол-во изображений для каждого символа

encoder = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # [1, 105, 105] -> [64, 105, 105]
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),  # [64, 105, 105] -> [64, 52, 52]

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [64, 52, 52] -> [128, 52, 52]
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),  # [128, 52, 52] -> [128, 26, 26]

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [1, 105, 105] -> [64, 105, 105]
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [64, 52, 52] -> [128, 52, 52]
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # [64, 52, 52] -> [128, 52, 52]
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten()
    )


def sample_episode(dataset, n_way=5, k_shot=1, n_query=5):
    """Создает эпизод для обучения Protonet."""
    # Выбираем только классы с достаточным количеством примеров
    valid_classes = [c for c in dataset.keys()
                     if len(dataset[c]) >= (k_shot + n_query)]
    classes = np.random.choice(valid_classes, n_way, replace=False)

    support = []
    query = []

    for class_id in classes:
        class_images = dataset[class_id]
        # Выбираем случайные индексы без повторений
        indices = np.random.choice(len(class_images),
                                   k_shot + n_query,
                                   replace=False)

        # Преобразуем в тензоры
        for i, idx in enumerate(indices[:k_shot]):
            img = torch.from_numpy(class_images[idx]).float()
            support.append(img.unsqueeze(0).unsqueeze(0))

        for idx in indices[k_shot:]:
            img = torch.from_numpy(class_images[idx]).float()
            query.append(img.unsqueeze(0).unsqueeze(0))

    # Собираем батчи
    support = torch.cat(support, dim=0) if support else torch.tensor([])
    query = torch.cat(query, dim=0) if query else torch.tensor([])

    return support, query, classes


n_way = 20  # Количество классов в эпизоде
k_shot = 1  # Количество примеров на класс в support set
n_query = 5  # Количество query-примеров на класс


class Protonet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support, query):
        """
        Args:
            support: [n_way * k_shot, C, H, W]
            query: [n_way * n_query, C, H, W]
        Returns:
            logits: [n_way * n_query, n_way]
        """
        # Получаем эмбеддинги
        support_emb = self.encoder(support)  # [n_way * k_shot, emb_dim]
        query_emb = self.encoder(query)  # [n_way * n_query, emb_dim]

        # Вычисляем прототипы (средние по классам)
        prototypes = support_emb.reshape(n_way, k_shot, -1).mean(1)  # [n_way, emb_dim]

        # Считаем расстояния query до прототипов
        dists = torch.cdist(query_emb, prototypes)  # [n_query * n_way, n_way]

        # Преобразуем расстояния в "логиты"
        return -dists


# Инициализация модели
protonet = Protonet(encoder)
optimizer = torch.optim.Adam(protonet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x_loss = []
y_loss = []
# Обучение Protonet
for epoch in range(100):
    protonet.train()

    try:
        support, query, classes = sample_episode(dataset, n_way, k_shot, n_query)
        if len(support) == 0 or len(query) == 0:
            continue

        # Формируем метки для query
        query_labels = torch.repeat_interleave(torch.arange(len(classes)), n_query)

        # Forward pass
        logits = protonet(support, query)
        loss = criterion(logits, query_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        x_loss.append(epoch)
        y_loss.append(loss.item())
    except ValueError as e:
        print(f"Skipping epoch {epoch}: {str(e)}")
        continue
plt.plot(x_loss, y_loss)
plt.title("loss во время обучения ProtoNet")
plt.xlabel("Эпоха")
plt.ylabel("loss")
plt.show()

torch.save(protonet, 'protonet_model.pth')  # Сохраняет всю модель
# Тестирование
protonet.eval()
correct = 0
total = 0

for _ in range(10):
    try:
        support, query, classes = sample_episode(testset, n_way, k_shot, n_query)
        if len(query) == 0:
            continue

        query_labels = torch.repeat_interleave(torch.arange(len(classes)), n_query)

        with torch.no_grad():
            logits = protonet(support, query)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == query_labels).sum().item()
            total += len(query_labels)
    except ValueError as e:
        print(f"Skipping test episode: {str(e)}")
        continue

if total > 0:
    print(f"Test Accuracy: {correct / total:.4f}")
else:
    print("No test episodes were processed")

