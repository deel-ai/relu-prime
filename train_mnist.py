import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from my_relu import ReLUAlpha
from tqdm import tqdm
import pandas as pd
from data_utils import get_mnist_loaders
from train import train, test


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init(alpha=0, regularization="relu"):
    if regularization == "relu":
        net = torch.torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 10),
        ).to(device)

    elif regularization == "batch_norm":
        net = torch.torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(alpha),
            nn.Linear(2048, 10),
        ).to(device)

    elif regularization == "dropout":
        net = torch.torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2048),
            ReLUAlpha(alpha),
            nn.Dropout(args["dropout_rate"]),
            nn.Linear(2048, 2048),
            ReLUAlpha(alpha),
            nn.Dropout(args["dropout_rate"]),
            nn.Linear(2048, 2048),
            ReLUAlpha(alpha),
            nn.Dropout(args["dropout_rate"]),
            nn.Linear(2048, 10),
        ).to(device)

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch relu experiment impact of different values for ReLU'(0)"
    )
    parser.add_argument("--relu", type=float, default="0")
    parser.add_argument("--epochs", type=int, default=3, help="nb epochs")
    parser.add_argument("--regularization", type=str, default="relu")
    parser.add_argument(
        "--nb_experiment",
        type=int,
        default=30,
        help="number of experiment independant run for each configuration",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning_rate",
    )
    parser.add_argument("--dropout_rate", type=float, default=0, help="dropout_rate")
    args = vars(parser.parse_args())

    outdir = f"./results/mnist_sgd"
    alpha = args["relu"]
    regularization = args["regularization"]

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if regularization == "relu":
        file_name = f"relu_{alpha}.pkl"
    elif regularization == "batch_norm":
        file_name = f"batch_norm_relu_{alpha}.pkl"
    elif regularization == "dropout":
        file_name = f'dropout_{args["dropout_rate"]}_relu_{alpha}.pkl'
    n_epochs = args["epochs"]
    nb_experiment = args["nb_experiment"]

    print(f"Running MNIST with ReLU'(0)={alpha} for {n_epochs} epochs")
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    trainloader, testloader = get_mnist_loaders()

    results_df = pd.DataFrame(
        columns=[
            "run_id",
            "epoch",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "relu",
        ]
    )

    for k in tqdm(range(nb_experiment), desc="run_loop", leave=False):
        net = init(alpha, regularization)
        optimizer = optim.SGD(
            net.parameters(),
            lr=args["learning_rate"],
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in tqdm(range(n_epochs), desc="epoch_loop", leave=False):
            train_loss, train_acc = train(net, optimizer, trainloader)
            test_loss, test_acc = test(net, testloader)
            results_df = results_df.append(
                {
                    "run_id": k,
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "test_accuracy": test_acc,
                    "train_accuracy": train_acc,
                    "relu": args["relu"],
                },
                ignore_index=True,
            )
            scheduler.step()

    path = os.path.join(outdir, file_name)
    results_df.to_pickle(path)
