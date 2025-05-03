import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1)

class ImageDataset(Dataset):
    '''
    Modified to load train/val/test images of post-disaster datasets from different folders
    '''
    def __init__(self, phase, dataset_name):
        self.phase = phase  # train/val/test
        self.dataset_name = dataset_name  # FloodNet/RescueNet
        if phase == "train":
            self.lr_root = os.path.join("datasets/uploads/New_LR_dataset_512_train", dataset_name, "LR/train-org-img")
            self.sr_root = os.path.join("datasets/uploads/New_LR_dataset_512_train", dataset_name, "HR/train-org-img")
        elif phase == "val":
            self.lr_root = os.path.join("datasets/uploads/New_LR_dataset_512_val", dataset_name, "LR/train-org-img")
            self.sr_root = os.path.join("datasets/uploads/New_LR_dataset_512_val", dataset_name, "HR/train-org-img")
        elif phase == "test":
            self.lr_root = os.path.join("datasets/uploads/New_LR_dataset_512_test", dataset_name, "LR/train-org-img")
            self.sr_root = os.path.join("datasets/uploads/New_LR_dataset_512_test", dataset_name, "HR/train-org-img")

        if not os.path.exists(self.lr_root):
            raise ValueError(f"Low resolution image directory {self.lr_root} does not exist.")
        if not os.path.exists(self.sr_root):
            raise ValueError(f"Super resolution image directory {self.sr_root} does not exist.")

        self.lr_files = sorted(glob.glob(os.path.join(self.lr_root, "*")))
        self.sr_files = sorted(glob.glob(os.path.join(self.sr_root, "*")))

        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __getitem__(self, index):
        lr_path = self.lr_files[index]
        hr_path = self.sr_files[index]
        img_lr = Image.open(lr_path).convert("RGB")
        img_hr = Image.open(hr_path).convert("RGB")

        img_lr = self.lr_transform(img_lr)
        img_hr = self.hr_transform(img_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.lr_files)
