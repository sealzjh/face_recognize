# -*- encoding: utf8 -*-
from config import DATA_TRAIN
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

    dataset = ImageFolder(root=DATA_TRAIN, transform=data_transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset_loader
