# -*- encoding: utf8 -*-

import config
import os
import torch
from torch import nn
from data_set import get_dataset, get_transform
from model import Net
import torch.nn.functional as F

# 是否使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    train_loader = get_dataset(batch_size=config.BATCH_SIZE)
    net = Net().to(DEVICE)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(config.EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            loss = loss_fun(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, step * len(x), len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))

    torch.save(net.state_dict(), os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL))

    return net


def predict_model(image):
    data_transform = get_transform()
    image = data_transform(image)
    image = image.view(-1, 3, 32, 32)
    net = Net().to(DEVICE)
    y_out= net(image.to(DEVICE))
    return F.softmax(y_out, dim=1).max(1, keepdim=True)[1][0][0].item()

