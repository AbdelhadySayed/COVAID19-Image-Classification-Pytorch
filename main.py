

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

from train import train
from eval import test
from data_loaders import create_data_loaders
from model import net



def main(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    model=net()
     model.cuda()
    model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_loader, valid_loader, test_loader, train_size, valid_size = create_data_loaders(args.data_dir, args.batch_size)
    epoch_times = []
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    for epoch in range(1, args.epochs+1):
        model, train_loss, val_loss, train_acc, val_acc, epoch_time=train(model, train_loader, valid_loader, train_size, valid_size, criterion,
         optimizer)

        test_loss, test_acc = test(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        #val_losses.append(val_loss)
        test_losses.append(test_loss)

        epoch_times.append(epoch_time)

        train_accs.append(train_acc)
        #val_accs.append(val_acc)
        test_accs.append(test_acc)
        print("-----------------------------------------------------------\n",
             "Epoch {}: train loss {:.3f}, test_loss {:.3f}, in {:.1f} sec\n".format(
                epoch, train_loss, test_loss, epoch_time)
         )
        print(
             "  train Accuracy {:.0f}%, test Accuracy {:.0f}%\n".format(
                train_acc, test_acc)
         )

    print("\n\n\n", train_losses, test_losses, epoch_times)
    print(train_accs, val_accs, test_accs)
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            "epoch_times": epoch_times,
            'train_accs': train_accs,
            "test_accs": test_accs,
            }, "/model.pth")

    epochs = [1:args.epochs]
    fig = plt.figure(figsize =(12, 5))
    fig.suptitle('Train and Validation Losses')
    ax = fig.add_subplot()
    bp = plt.plot(epochs, train_accs, color='orange', label='Train Accuracy', lw=1)
    bp = plt.plot(epochs, test_accs, color='blue', label='Test Accuracy', lw=1)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.legend(loc=4, prop={'size': 12})
    plt.show()

    fig = plt.figure(figsize =(10, 5))
    fig.suptitle('Train and Validation Losses')
    ax = fig.add_subplot()
    bp = plt.plot(epochs, train_losses, color='orange', label='Train Loss', lw=1)
    bp = plt.plot(epochs, test_losses, color='blue', label='Test Loss', lw=1)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.legend(loc=4, prop={'size': 12})
    plt.show()








if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",   # the actual variable is batch_size
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
#    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 25)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str,
        default="./COVID19-DATASET",
        metavar="DD",
        help="data directory",
    )

    args = parser.parse_args()
    main(args)