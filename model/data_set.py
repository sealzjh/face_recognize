# -*- encoding: utf8 -*-
import config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transform():
    return transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])


def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()

    train_dataset = ImageFolder(root=config.DATA_TRAIN, transform=data_transform)
    test_dataset = ImageFolder(root=config.DATA_TEST, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader
