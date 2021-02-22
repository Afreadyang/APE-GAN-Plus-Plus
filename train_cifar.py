# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import TensorDataset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import Generator, Discriminator, CifarCNN

from torch.utils.tensorboard import SummaryWriter


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def show_images(e, x, x_adv, x_fake, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].set_title("Normal")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[1, i].imshow(x_adv[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_fake[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[2, i].imshow(x_fake[i, 0].cpu().numpy(), cmap="gray")
        axes[2, i].set_title("Proposed_GAN")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))


def main(args):
    lr = args.lr
    epochs = args.epochs
    batch_size = 128

    target_model = CifarCNN().cuda()
    target_model.load_state_dict(torch.load("./CNN/cifar_cnn.tar")["state_dict"])
    target_model.eval()

    check_path = args.checkpoint
    os.makedirs(check_path, exist_ok=True)
    fgsm = torch.load('./CIFAR10_AE/Training_Set/fgsm.tar')
    train_data = fgsm['normal_data_x']
    train_label = fgsm['normal_data_t']
    adv_fgsm = fgsm['adv']

    x_tmp = train_data[190000:190005]
    x_adv_tmp = adv_fgsm[190000:190005]

    train_data = TensorDataset(train_data,train_label,adv_fgsm)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    in_ch = 1 if args.data == "mnist" else 3
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    G = Generator(in_ch).cuda()
    D = Discriminator(in_ch).cuda()

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    cudnn.benchmark = True
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    writer = SummaryWriter('./log/cifar')
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 2)
    print("\t".join(["{:}"] * 3).format("Epoch", "Gen_Loss", "Dis_Loss"))
    for e in range(epochs):
        G.eval()
        x_fake = G(Variable(x_adv_tmp.cuda())).data
        show_images(e, x_tmp, x_adv_tmp, x_fake, check_path)
        G.train()
        gen_loss, dis_loss, n = 0, 0, 0
        for x, x_label, x_adv in tqdm(train_loader, total=len(train_loader), leave=False):
            x, x_real_label, x_adv = Variable(x.cuda()),Variable(x_label.cuda()),Variable(x_adv.cuda())
            x_real_label = Variable(x_real_label.type(LongTensor))
            # Train D
            y_real = D(x).squeeze()
            x_fake = G(x_adv)
            y_fake = D(x_fake).squeeze()
            gradient_penalty = compute_gradient_penalty(D, x.data, x_fake.data)
            lambda_gp = 10
            loss_D = -torch.mean(y_real) + torch.mean(y_fake) + lambda_gp * gradient_penalty 
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G
            for _ in range(1):
                x_fake = G(x_adv)
                y_fake = D(x_fake).squeeze()
                # if e >100:
                loss_G = -torch.mean(y_fake) + 0.5*auxiliary_loss(target_model(x_fake),x_real_label) + 0.5*loss_mse(x_fake, x)
                # else:
                	# loss_G = -torch.mean(y_fake)
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            gen_loss += loss_D.item() * x.size(0)
            dis_loss += loss_G.item() * x.size(0)
            n += x.size(0)
        writer.add_scalar('dis_loss', dis_loss, e)
        writer.add_scalar('gen_loss', gen_loss, e)
        print(print_str.format(e, gen_loss / n, dis_loss / n))
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()},
                   os.path.join(check_path, "{}.tar".format(e + 1)))

    G.eval()
    x_fake = G(Variable(x_adv_tmp.cuda())).data
    show_images(epochs, x_tmp, x_adv_tmp, x_fake, check_path)
    G.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/cifar")
    args = parser.parse_args()
    main(args)
