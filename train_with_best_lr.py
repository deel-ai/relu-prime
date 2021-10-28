import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from resnet import resnet18
from vgg import VGG
from my_relu import ReLUAlpha
from data_utils import get_cifar10_loaders, get_svhn_loaders, get_mnist_loaders
from train import train, test


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init(network="vgg11", batch_norm=False, alpha=0, optimizer="sgd", lr=0.01):
    if network == "vgg11":
        net = VGG("VGG11", batch_norm=batch_norm, relu_fn=lambda: ReLUAlpha(alpha)).to(
            device
        )
    elif network == "resnet18":
        if batch_norm:
            net = resnet18(
                norm_layer=nn.BatchNorm2d, relu_fn=lambda: ReLUAlpha(alpha)
            ).to(device)
        else:
            net = resnet18(norm_layer=nn.Identity, relu_fn=lambda: ReLUAlpha(alpha)).to(
                device
            )
    elif network == "mnist":
        if batch_norm:
            norm_layer = lambda: nn.BatchNorm1d(2048)

        else:
            norm_layer = lambda: nn.Identity

        net = torch.torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2048),
            norm_layer(),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            norm_layer(),
            ReLUAlpha(alpha),
            nn.Linear(2048, 2048),
            norm_layer(),
            ReLUAlpha(alpha),
            nn.Linear(2048, 10),
        ).to(device)

    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)
    return net, opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch relu experiment impact of different values for ReLU'(0) with best lr"
    )
    parser.add_argument("--epochs", type=int, default=200, help="nb epochs")
    parser.add_argument(
        "--nb_experiment",
        type=int,
        default=5,
        help="number of experiment independant run for each configuration",
    )
    parser.add_argument("--batch_norm", type=boolean_string, default="True")
    parser.add_argument("--network", type=str, default="mnist")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--optimizer", type=str, default="sgd")

    args = vars(parser.parse_args())
    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    batch_norm = args["batch_norm"]
    nb_experiment = args["nb_experiment"]

    outdir = (
        f"./results/section4/{dataset}/{network}/batch_norm_{batch_norm}/{optimizer}"
    )

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    file_name = f"results.pkl"

    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "svhn":
        trainloader, testloader = get_svhn_loaders()
    elif dataset == "mnist":
        trainloader, testloader = get_mnist_loaders()

    best_lrs = pd.read_csv(
        f"results/best_learning_rates/{dataset}//{network}/batch_norm_{batch_norm}/{optimizer}/best_lr.csv"
    )
    relu_values = best_lrs.relu.values
    learning_rates = best_lrs.lr.values

    results_df = pd.DataFrame(
        columns=[
            "run_id",
            "epoch",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "relu",
            "lr",
        ]
    )
    for relu_value, lr in tqdm(
        zip(relu_values, learning_rates), desc="relu_value", leave=False
    ):
        for k in tqdm(range(nb_experiment), desc="run_loop", leave=False):
            net, opt = init(network, batch_norm, relu_value, optimizer, lr)
            for epoch in tqdm(range(epochs), desc="epoch_loop", leave=False):
                train_loss, train_acc = train(net, opt, trainloader)
                test_loss, test_acc = test(net, testloader)
                results_df = results_df.append(
                    {
                        "run_id": k,
                        "epoch": epoch,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                        "test_accuracy": test_acc,
                        "train_accuracy": train_acc,
                        "relu": relu_value,
                        "lr": lr,
                    },
                    ignore_index=True,
                )

    path = os.path.join(outdir, file_name)
    results_df.to_pickle(path)
