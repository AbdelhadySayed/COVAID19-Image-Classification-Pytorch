
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.optim as optim




def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    # print(
    #     "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
    #         test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
    #     )
    # )
    return test_loss, test_acc
    

    