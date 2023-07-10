
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from random import shuffle
import os


def create_data_loaders(data_dir, batch_size, valid_size=0.2):
    

    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }
    
    tr_dir = os.path.join(data_dir, 'train')
    train_data = torchvision.datasets.ImageFolder(root=tr_dir, transform=transform['train'])
    val_data = torchvision.datasets.ImageFolder(root=tr_dir, transform=transform['test'])

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    traindata_size = {"train":len(train_idx), "val":len(val_idx)}
    train_sampler = SubsetRandomSampler(train_idx) # Sampler for splitting train and val images
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size) 
    valid_loader = torch.utils.data.DataLoader(val_data,
                   sampler=val_sampler, batch_size=batch_size)
    
    dataloaders = {"train":train_loader, "val":valid_loader}
    data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val']}
    train_size = data_sizes["train"]
    valid_size = data_sizes["val"]

    tst_dir = os.path.join(data_dir, 'test')
    testdata = torchvision.datasets.ImageFolder(root=tst_dir, transform=transform['test'])
    test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, train_size, valid_size