import torchvision
import torchvision.transforms as transforms
import torch

def get_cifar10_train_loader():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

def get_cifar10_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

def get_cifar10_loaders():
    trainloader = get_cifar10_train_loader()
    testloader = get_cifar10_test_loader()
    return trainloader, testloader

def get_svhn_loaders():
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader

def get_mnist_loaders(**kwargs):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader