import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pandas as pd

from resnet import resnet18
from vgg import VGG
from relu import ReLUAlpha
from data_utils import *


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init(batch_norm=False, relu=0, optimizer="sgd", lr=0.1):
    if batch_norm:
        net = torch.torch.nn.Sequential(
            nn.Linear(28 * 28, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 10),
        ).to(device)

    else:
        net = torch.torch.nn.Sequential(
            nn.Linear(28 * 28, 2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 2048),
            ReLUAlpha(relu),
            nn.Linear(2048, 10),
        ).to(device)

    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)
    return net, opt


def train(net, optimizer, trainloader):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    bar = tqdm(iter(trainloader), desc="batch_loop", leave=False)
    for inputs, targets in bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs.view(-1, 28 * 28))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_description("accuracy %0.2f" % (100 * correct / total))
    return train_loss / len(trainloader), correct / total


def test(testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bar = tqdm(iter(testloader), desc="batch_loop", leave=False)
        for inputs, targets in bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs.view(-1, 28 * 28))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            bar.set_description("accuracy %0.2f" % (100 * correct / total))

    return test_loss / len(testloader), correct / total


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
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--optimizer", type=str, default="sgd")

    args = vars(parser.parse_args())
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    batch_norm = args["batch_norm"]
    nb_experiment = args["nb_experiment"]

    outdir = f"./results/section4/MNIST/{optimizer}/batch_norm_{batch_norm}"

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    file_name = f"results.pkl"

    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader, testloader = get_mnist_loaders()

    best_lrs = pd.read_csv(
        f"results/best_learning_rates/MNIST_v3times/batch_norm_{batch_norm}/sgd/best_lr.csv"
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
            net, opt = init(batch_norm, relu_value, optimizer, lr)
            for epoch in tqdm(range(epochs), desc="epoch_loop", leave=False):
                train_loss, train_acc = train(net, opt, trainloader)
                test_loss, test_acc = test(testloader)
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
