from Model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(dataloader, len_input, len_output, model, optim, criterion, epoch, device):
    sum_loss = 0
    avg_loss = 0
    model.train()
    for i, (X,y) in enumerate(dataloader):

        i = i+1
        # X = X.permute(1,0,2)
        # y = y.permute(1,0,2)
        X = X.to(device)
        y = y.to(device)

        pred = model.forward(X, y, y_mask= None)
        loss = criterion(pred, y[:,:,0].unsqueeze(-1))
#         print('Pred:\t', pred) 
#         print('Target:\t', y[:,:,0].unsqueeze(-1))
#         print('Loss:\t', loss)
#         print('Check loss:\t', (pred.item()- y[0][0][0].item())**2)
#         break
        optim.optimizer.zero_grad()
        loss.backward()
        optim.step()

        sum_loss += loss.item()
        avg_loss = sum_loss / i

        # if i%1000 == 0:
        #     print('Epoch: [{}][{}/{}]\t\tLoss: {:.3f}e-3'.format(epoch+1, i, len(dataloader), avg_loss*1000))

        if i == len(dataloader)-(len_input+len_output):
            print('Training Epoch [{}]\t\tLoss: {:.3f}e-3'.format(epoch+1, avg_loss*1000))
            break
    
    return avg_loss
