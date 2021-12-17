import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
from DataLoader import Dataset
from Model import Transformer, AdamWarmup
import torch.nn as nn
from joblib import load


def eval(val_loader, len_input, len_output, model, scaler, criterion, device):
    pred_hist = []
    loss_hist = []
    
    model.eval()
    for i, (X,y) in enumerate(val_loader):
        i = i+1
        X = X.to(device)
        y = y.to(device)

        pred = model.forward(X, y, y_mask = None)
        pred_hist.append(pred.item())
        loss = criterion(pred, y[:,:,0].unsqueeze(-1))
        loss_hist.append(loss.item())

        if i == len(val_loader)-(len_input+len_output):
            print('Prediction ended')
            break
    
    # pred_hist = scaler.inverse_transform(pred_hist)

    return pred_hist, loss_hist