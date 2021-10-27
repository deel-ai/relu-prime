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
from my_relu import ReLUAlpha
from data_utils import get_cifar10_loaders, get_svhn_loaders


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init():
    if args["network"] == "vgg11":
        if args["batch_norm"]:
            net = VGG(
                "VGG11", batch_norm=True, relu_fn=lambda: ReLUAlpha(args["relu"])
            ).to(device)
        else:
            net = VGG("VGG11", relu_fn=lambda: ReLUAlpha(args["relu"])).to(device)

    elif args["network"] == "vgg16":
        net = VGG("VGG16", relu_fn=lambda: ReLUAlpha(args["relu"])).to(device)
    elif args["network"] == "resnet18":
        if args["batch_norm"]:
            net = resnet18(
                norm_layer=nn.BatchNorm2d, relu_fn=lambda: ReLUAlpha(args["relu"])
            ).to(device)
        else:
            net = resnet18(
                norm_layer=nn.Identity, relu_fn=lambda: ReLUAlpha(args["relu"])
            ).to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args["optimizer"] == "sgd":
        optimizer = optim.SGD(
            net.parameters(), lr=args["learning_rate"], momentum=0.9, weight_decay=5e-4
        )
    elif args["optimizer"] == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    return net, optimizer, scheduler


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
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_description("accuracy %0.2f" % (100 * correct / total))
    return train_loss / len(trainloader), correct / total


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bar = tqdm(iter(testloader), desc="batch_loop", leave=False)
        for inputs, targets in bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            bar.set_description("accuracy %0.2f" % (100 * correct / total))

    return test_loss / len(testloader), correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch relu experiment impact of different values for ReLU'(0)"
    )
    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--relu", type=float, default="0")
    parser.add_argument("--epochs", type=int, default=10, help="nb epochs")
    parser.add_argument(
        "--nb_experiment",
        type=int,
        default=3,
        help="number of experiment independant run for each configuration",
    )
    parser.add_argument("--batch_norm", type=boolean_string, default="True")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="learning_rate"
    )
    args = vars(parser.parse_args())

    outdir = f'./results/{args["network"]}_{args["dataset"]}_{args["optimizer"]}'

    if args["batch_norm"]:
        outdir += "_batch_norm"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    file_name = f'relu_{args["relu"]}.pkl'
    n_epochs = args["epochs"]
    nb_experiment = args["nb_experiment"]

    print(
        f"Running {args['network']} with ReLU'(0)={args['relu']} for {n_epochs} epochs on CIFAR10"
    )
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args["dataset"] == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif args["dataset"] == "svhn":
        trainloader, testloader = get_svhn_loaders()

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
        net, optimizer, scheduler = init()
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
