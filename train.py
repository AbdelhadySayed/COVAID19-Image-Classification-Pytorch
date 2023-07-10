
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.optim as optim
import time
import copy



def train(model, train_loader, valid_loader, train_size, valid_size, criterion, optimizer):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    
    start = time.time()
    model.train() 
    train_loss = 0
    val_loss = 0
    correct_tr = 0 
    correct_val = 0
    with torch.set_grad_enabled(mode=True):
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_tr += pred.eq(targets.view_as(pred)).sum().item()
    train_acc = 100.0 * correct_tr /train_size 
    train_loss /= train_size
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(valid_loader):
            #inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_val += pred.eq(targets.view_as(pred)).sum().item()
    val_loss /= valid_size
    val_acc = 100.0 * correct_val /valid_size        
    epoch_time = time.time() - start
    model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_wts)
    return model, train_loss, val_loss, train_acc, val_acc, epoch_time 