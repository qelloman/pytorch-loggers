import sys
import tqdm

from sklearn.model_selection import train_test_split

import torch
import wandb

sys.path.append("../")

from models.data import MNISTDataSet
from models.mnist import MNISTPredictor
from models.utils import (
    set_seed,
    validate_model,
    get_dataloaders,
    loss,
)

train_valid_dataset = MNISTDataSet("./data/mnist_train.csv")
test_dataset = MNISTDataSet("./data/mnist_test.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shape = (1, 28, 28)
n_class = 10


def train():
    wandb.init()
    config = wandb.config

    set_seed(config.seed)
    train_dataset, valid_dataset = train_test_split(
        train_valid_dataset, test_size=0.2, random_state=config.seed, shuffle=True
    )
    model = MNISTPredictor(shape, config.latent_dim, n_class=n_class)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=0.0001,
    )
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        train_dataset, valid_dataset, test_dataset, config.batch_size
    )

    for epoch in range(config.max_epoch):
        model.train()
        train_loss, acc_cnt, n = 0.0, 0, 0

        for X, y_val in tqdm.tqdm(train_dataloader, ncols=50):
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
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l.cpu().item()
            n += X.shape[0]
            acc_cnt += torch.sum(y_pred == y_val)

        train_loss /= n
        acc = acc_cnt / n

        # Log loss and accuracy.
        training_metrics = {
            "train/loss": train_loss,
            "train/acc": acc,
            "train/epoch": epoch,
        }
        valid_loss, valid_acc = validate_model(model, valid_dataloader, n_class)
        valid_metrics = {
            "valid/loss": valid_loss,
            "valid/acc": valid_acc,
            "valid/epoch": epoch,
        }
        wandb.log({**training_metrics, **valid_metrics})

        if (epoch + 1) % 5 == 0:
            # log how the parameters change over training steps
            for name, weight in model.named_parameters():
                continue


sweep_config = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "valid_loss"},
    "parameters": {
        "batch_size": {"values": [128, 256]},
        "lr": {"values": [1e-2, 1e-3]},
        "latent_dim": {"values": [8, 16]},
        "max_epoch": {"value": 50},
        "seed": {"value": 0},
    },
}

sweep_id = wandb.sweep(sweep_config, project="my-second-sweep")
wandb.agent(sweep_id, function=train)
