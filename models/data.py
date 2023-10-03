import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class MNISTDataSet(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.x = torch.from_numpy(self.data[:, 1:]).view(-1, 1, 28, 28).float() / 255.0
        self.y = torch.from_numpy(self.data[:, [0]])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def show(self, index):
        x, y = self.__getitem__(index)
        x = x.view(28, 28) / 255.0
        plt.imshow(x, cmap="gray_r")
