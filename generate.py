# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

from models import MnistCNN, CifarCNN
from utils import accuracy, fgsm


def load_dataset(args):
    if args.data == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root = './', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root = './', train=False, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    elif args.data == "cifar":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './', train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    return train_loader, test_loader


def main(args):
    train_loader, test_loader = load_dataset(args)
    model = CifarCNN().cuda()
    # model = MnistCNN().cuda()
    model.load_state_dict(torch.load("./CNN/cifar_cnn.tar")["state_dict"])
    # model.load_state_dict(torch.load("./CNN/mnist_cnn.tar")["state_dict"])
    model.eval()
    loss_func = nn.CrossEntropyLoss().cuda()
    
    # Generate Adversarial Examples
    print("-" * 30)
    print("Genrating Adversarial Examples ...")
    eps = args.eps
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data_x, normal_data_t, adv_data = None, None, None
    for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
        x, t = Variable(x.cuda()), Variable(t.cuda())
        y = model(x)
        train_acc += accuracy(y, t)

        x_adv = fgsm(model, x, t, loss_func, eps)
        y_adv = model(x_adv)
        adv_acc += accuracy(y_adv, t)
        train_n += t.size(0)

        x, t, x_adv = x.data, t.data, x_adv.data
        if normal_data_x is None:
            normal_data_x, normal_data_t, adv_data = x, t, x_adv
        else:
            normal_data_x = torch.cat((normal_data_x, x))
            normal_data_t = torch.cat((normal_data_t, t))
            adv_data = torch.cat((adv_data, x_adv))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(train_acc / train_n * 100, adv_acc / train_n * 100))
    torch.save({"normal_data_x": normal_data_x, "normal_data_t":normal_data_t, "adv": adv_data}, "data.tar")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--eps", type=float, default=0.15)
    args = parser.parse_args()
    main(args)
