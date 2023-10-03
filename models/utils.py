import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


@torch.no_grad()
def draw_images(model, latent_dim, n_img=4):
    model.eval()
    device = next(model.parameters()).device
    z = torch.randn(n_img, latent_dim).to(device)
    images = model.decoder(z)
    img_grid = make_grid(images, nrow=n_img)

    return img_grid


@torch.no_grad()
def test_samples(model, dataloader, n_samples=10):
    X_batch, y_val_batch = next(iter(dataloader))
    X_samples, y_val_samples = X_batch[:n_samples], y_val_batch[:n_samples]

    model.eval()
    y_pred, y_prob = model(X_samples)

    img_grid = make_grid(X_samples, nrow=n_samples)
    return y_pred, y_val_samples, img_grid


def get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return train_dataloader, valid_dataloader, test_dataloader


bce_loss = nn.BCELoss(reduction="sum")


def loss(y_pred, y_true):
    loss = bce_loss(y_pred, y_true)
    return loss


def validate_model(model, dataloader, n_class):
    model.eval()
    device = next(model.parameters()).device
    valid_loss = 0.0
    acc_cnt = 0
    n = 0
    with torch.inference_mode():
        for X, y_val in tqdm.tqdm(dataloader, ncols=50):
            X = X.to(device)
            y_val = y_val.to(torch.int64).to(device)

            y_prob, y_pred = model(X)
            y_true = (
                torch.nn.functional.one_hot(y_val, num_classes=n_class)
                .squeeze()
                .float()
                .to(device)
            )

            l = loss(y_prob, y_true).to(device)
            valid_loss += l.cpu().item()
            n += X.shape[0]
            acc_cnt += torch.sum(y_pred == y_val)

        valid_loss /= n
        valid_acc = acc_cnt / n
        return valid_loss, valid_acc


class ArgsDict:
    def __init__(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)

    def update(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)
