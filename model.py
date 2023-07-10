import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def net():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights="IMAGENET1K_V2") 
    ct = 0
    for child in model.children():
        ct += 1
        if ct <38:
            for param in child.parameters():
                param.requires_grad = False 
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2))
    model = model.to(device)
    return model