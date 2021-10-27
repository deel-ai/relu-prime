import torch
import torch.nn as nn
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


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
