import torch

def euclidean_dist(x, y):
    """
    Расчет евклидова рсстояния каждого элемента до каждого центроида
    Args:
        x (torch.Tensor): shape (n, d).
        y (torch.Tensor): shape (m, d).
    :return:
    тензор расстояний - от каждого элемента до каждого центроида - shape (n, m)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)