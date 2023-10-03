import sys
import os
import datetime
import tqdm
from itertools import product

from sklearn.model_selection import train_test_split

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../")

from models.data import MNISTDataSet
from models.mnist import MNISTPredictor
from models.utils import (
    set_seed,
    validate_model,
    get_dataloaders,
    loss,
    ArgsDict,
)

train_valid_dataset = MNISTDataSet("./data/mnist_train.csv")
test_dataset = MNISTDataSet("./data/mnist_test.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shape = (1, 28, 28)
n_class = 10


def train(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{args.log_root_dir}/{current_time}"
    tb = SummaryWriter(log_dir=log_dir)

    set_seed(args.seed)
    train_dataset, valid_dataset = train_test_split(
        train_valid_dataset, test_size=0.2, random_state=args.seed, shuffle=True
    )
    model = MNISTPredictor(shape, args.latent_dim, n_class=n_class)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.0001,
    )
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        train_dataset, valid_dataset, test_dataset, args.batch_size
    )

    for epoch in range(args.max_epoch):
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
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("train/acc", acc, epoch)

        valid_loss, valid_acc = validate_model(model, valid_dataloader, n_class)
        tb.add_scalar("valid/loss", valid_loss, epoch)
        tb.add_scalar("valid/acc", valid_acc, epoch)

        if (epoch + 1) % 5 == 0:
            # log how the parameters change over training steps
            for name, weight in model.named_parameters():
                tb.add_histogram(name, weight, epoch + 1)
                tb.add_histogram(f"{name}.grad", weight.grad, epoch + 1)

            # log y_val and y_pred with the actual figures

    test_loss, test_acc = validate_model(model, test_dataloader, n_class)

    # we can check performance across different hyperparameters
    # NOTE: if run_name is not specified this way, then it will generate seperate dirs to store metrics for hparams.
    tb.add_hparams(
        {"lr": args.lr, "bsize": args.batch_size, "latent_dim": args.latent_dim},
        {"test/loss": test_loss, "test/acc": test_acc},
        run_name=os.path.dirname(os.path.realpath(__file__)) + os.sep + tb.log_dir,
    )

    # tensorboard draws the model's graph structure.
    # NOTE: add_graph requires input shape to the model.
    tb.add_graph(model, next(iter(test_dataloader))[0][[0]].to(device))
    tb.close()


parameters = dict(
    batch_size=[128, 256],
    lr=[1e-2, 1e-3],
    latent_dim=[4, 8, 16],
    # batch_size=[256],
    # lr=[1e-3],
    # latent_dim=[8],
)

param_values = [v for v in parameters.values()]

args = ArgsDict({"seed": 0, "max_epoch": 50, "log_root_dir": "mnist-tb"})

for run_idx, (batch_size, lr, latent_dim) in enumerate(product(*param_values)):
    args.update({"batch_size": batch_size, "lr": lr, "latent_dim": latent_dim})
    train(args)
