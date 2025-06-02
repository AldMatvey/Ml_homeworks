import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = InceptionResnetV1(pretrained='vggface2')

    def forward(self, inp):
        return self.encode(inp)
