import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from resnet32 import resnet32_normal as resnet32
import argparse

import numpy as np


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', default=16, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
parser.add_argument('--num_epochs', default=70, help='')
parser.add_argument('--num_models', default=3, help='')

def train(net, optimizer, step_lr_scheduler, train_loader, epoch):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
		
    acc = 100 * correct / total
    print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), acc))
    
    step_lr_scheduler.step()

def test(net, epoch):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
	
    acc = 100 * correct / total
    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
            epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc))

    return acc, test_loss/(batch_idx+1)


if __name__=='__main__':
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_test = CIFAR10(root='data', train=False, download=True, transform=transforms_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_worker)

    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)

    indices = list(range(args.num_models))
    DATA_SIZE_PER_CLASS = 5000
    subset_size = DATA_SIZE_PER_CLASS // args.num_models

    for i in range(10):
        temp_idx = np.where(np.array(dataset_train.targets) == i)[0]
        if i == 0:
            for j in range(args.num_models):
                indices[j] = temp_idx[j*subset_size:(j+1)*subset_size]
        for j in range(args.num_models):
            indices[j] = np.concatenate((indices[j], temp_idx[j*subset_size:(j+1)*subset_size]))

    subsets = [Subset(dataset_train, indice) for indice in indices]
    
    train_loaders = [DataLoader(subset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True) for subset in subsets]

    print('==> Making model..')


    criterion = nn.CrossEntropyLoss()
    decay_epoch = [20, 40, 50]
    num_epochs = args.num_epochs

    for net_num in range(args.num_models):
        net = resnet32().to(device)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
        train_loader = train_loaders[net_num]

        best_loss = 1000000
        best_acc = -1

        for epoch in range(1, num_epochs+1):
            train(net, optimizer, step_lr_scheduler, train_loader, epoch)
            acc, loss = test(net, epoch)
            if acc > best_acc:
                best_acc = acc
            if loss < best_loss:
                best_loss = loss
                state = {
                    'net': net.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch,
                }
                torch.save(state, 'pretrained_models/'+str(args.num_models)+'-models/resnet32_cifar10_split_'+str(net_num)+'.pth')

        print(net_num+1, "training finished!")