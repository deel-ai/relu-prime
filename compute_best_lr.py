import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from vgg import VGG
from resnet import resnet18
from relu import ReLUAlpha
from data_utils import *
from train import train, test
import optuna
from optuna.trial import TrialState
import joblib

NTRIALS = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init(network="vgg11", batch_norm=False, alpha=0):
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

    return net


def objective(trial):
    global optimizer
    global dataset
    global epochs
    global alpha
    global batch_norm

    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "svhn":
        trainloader, testloader = get_svhn_loaders()
    elif dataset == "mnist":
        trainloader, testloader = get_mnist_loaders()

    net = init(network, batch_norm, alpha)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train(net, opt, trainloader)
        test_loss, test_acc = test(net, testloader)
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lr search for ReLU'(0)")
    parser.add_argument("--relu", type=float, default="0")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--batch_norm", type=boolean_string, default="True")
    parser.add_argument("--epochs", type=int, default=1, help="nb epochs")
    parser.add_argument("--trials", type=int, default=2, help="nb epochs")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.1])
    args = vars(parser.parse_args())

    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    n_trials = args["trials"]
    batch_norm = args["batch_norm"]
    alphas = args["alphas"]

    outdir = f"./results/best_learning_rates/{dataset}/{network}/batch_norm_{batch_norm}/{optimizer}"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    file_name = f"best_lr_tmp.csv"
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    df = pd.DataFrame(columns=["relu", "lr"])

    for alpha in alphas:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        df = df.append({"relu": alpha, "lr": trial.params["lr"]}, ignore_index=True)
        study_path = os.path.join(outdir, "study.pkl")
        joblib.dump(study, study_path)
    csv_path = os.path.join(outdir, file_name)
    df.to_csv(csv_path)
