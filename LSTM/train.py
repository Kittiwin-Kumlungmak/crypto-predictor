from Model import LSTM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import Dataset


def train(train_loader, val_loader, state_in, len_input, d_model, len_output, model, optim, criterion, epoch, device):
    sum_train_loss = 0
    avg_train_loss = 0
    sum_val_loss = 0
    avg_val_loss = 0
    
    # Training Loop
    model.train()
    for i, (X_train,y_train) in enumerate(train_loader):

        i += 1
        # X = X.permute(1,0,2)
        # y = y.permute(1,0,2)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
 
        train_pred, _ = model.forward(X_train, state_in)
        train_loss = criterion(train_pred.view(1), y_train.squeeze(0).squeeze(0)[0].unsqueeze(-1))
#         print('Pred:\t', pred) 
#         print('Target:\t', y[:,:,0].unsqueeze(-1))
#         print('Loss:\t', loss)
#         print('Check loss:\t', (pred.item()- y[0][0][0].item())**2)
#         break
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        sum_train_loss += train_loss.item()
        avg_train_loss = sum_train_loss / i

        if i == len(train_loader)-(len_input+len_output):
            # print('Training Epoch [{}]\t\tLoss: {:.3f}e-3'.format(epoch+1, avg_train_loss*1000))
            break

    # Validation loop
    model.eval()
    for i, (X_val,y_val) in enumerate(val_loader):

        i += 1
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        val_pred, _ = model.forward(X_train, state_in)
        val_loss = criterion(val_pred.view(1), y_val.squeeze(0).squeeze(0)[0].unsqueeze(-1))

        sum_val_loss += val_loss.item()
        avg_val_loss = sum_val_loss / i

        # if i%1000 == 0:
        #     print('Epoch: [{}][{}/{}]\t\tLoss: {:.3f}e-3'.format(epoch+1, i, len(train_loader), avg_loss*1000))

        if i == len(val_loader)-(len_input+len_output):
            break
    
    return avg_train_loss, avg_val_loss
