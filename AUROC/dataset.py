from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


class MaskDataset(Dataset):

    def __init__(self, file_path, train=True, transform=None):
        self.file_path = file_path
        self.train = train
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.images = []
        self.labels = []
        self.type = "train" if train else "val"

        with_mask_path = os.path.join(self.file_path, self.type, "with_mask")
        for filename in os.listdir(with_mask_path):
            img = Image.open(os.path.join(with_mask_path, filename)).convert("RGB")
            img = self.transform(img)
            self.images.append(img)
            self.labels.append(1)
        without_mask_path = os.path.join(self.file_path, self.type, "without_mask")
        for filename in os.listdir(with_mask_path):
            img = Image.open(os.path.join(with_mask_path, filename)).convert("RGB")
            img = self.transform(img)
            self.images.append(img)
            self.labels.append(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class SynDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.images = []
        self.labels = []

        with_mask_path = os.path.join(self.file_path, "with_mask")
        for filename in os.listdir(with_mask_path):
            img = Image.open(os.path.join(with_mask_path, filename)).convert("RGB")
            img = self.transform(img)
            self.images.append(img)
            self.labels.append(1)
        without_mask_path = os.path.join(self.file_path, "without_mask")
        for filename in os.listdir(with_mask_path):
            img = Image.open(os.path.join(with_mask_path, filename)).convert("RGB")
            img = self.transform(img)
            self.images.append(img)
            self.labels.append(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


if __name__ == "__main__":
    dataset = MaskDataset(file_path="dataset/", train=False)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_loader))
