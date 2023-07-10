import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
    #model.summary()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights="IMAGENET1K_V2") # Returns Defined Densenet model with weights trained on ImageNet
    #model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')

    # for param in model.parameters():
    #     param.requires_grad = False   

    # num_features=model.fc.in_features
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.fc = nn.Sequential(
    #                nn.Linear(num_features, 120))
    #num_ftrs = model.classifier.in_features
    #for param in model.parameters():
        #param.requires_grad = False
    #model_ft = models.resnet50(pretrained=True)   
    #final_layer = list(model.children())[-1]    
    #num_features = model.classifier.in_features
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
    #model = model.to(device)
    return model