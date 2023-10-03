import torch
import torch.nn as nn
from collections import OrderedDict


device = "cuda" if torch.cuda.is_available else "cpu"


class MLP(nn.Module):
    def __init__(self, dims, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(dims) - 2) or ((i == len(dims) - 2) and last_activation):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU()))
            self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, shape, latent_dim=16):
        super(Encoder, self).__init__()
        c, w, h = shape
        ww = ((w - 8) // 2 - 4) // 2
        hh = ((h - 8) // 2 - 4) // 2
        self.encode = nn.Sequential(
            nn.Conv2d(c, 16, 5, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            MLP([ww * hh * 64, 256, 128, latent_dim]),
        )

    def forward(self, x):
        h = self.encode(x)
        return h


class MNISTPredictor(nn.Module):
    def __init__(self, shape, latent_dim, n_class):
        super().__init__()
        self.encoder = Encoder(shape, latent_dim)
        self.predictor = nn.Sequential(MLP([latent_dim, n_class]), nn.Softmax(dim=-1))

    def forward(self, x):
        h = self.encoder(x)
        y_prob = self.predictor(h)
        y_pred = torch.argmax(y_prob, dim=-1, keepdim=True)
        return y_prob, y_pred
