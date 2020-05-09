# -*- encoding: utf8 -*-

import config
import os
import torch
from data_set import get_dataset, get_transform
from model import Net
import torch.nn.functional as F

# 使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    train_loader, test_loader = get_dataset(batch_size=config.BATCH_SIZE)
    net = Net().to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(config.EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            loss = F.nll_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 3 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, step * len(x), len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))

    test(net, test_loader)

    torch.save(net.state_dict(), os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL))

    return net


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            test_loss += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print '\ntest loss={:.4f}, accuracy={:.4f}\n'.format(test_loss, float(correct) / len(test_loader.dataset))


def predict_model(image):
    data_transform = get_transform()
    image = data_transform(image)
    image = image.view(-1, 3, 32, 32)
    net = Net().to(DEVICE)
    # 加载模型参数权重
    net.load_state_dict(torch.load(os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL)))
    output = net(image.to(DEVICE))
    pred = output.max(1, keepdim=True)[1]
    return pred.item()

