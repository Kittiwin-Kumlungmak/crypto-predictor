import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
from DataLoader import Dataset
from Model import Transformer, AdamWarmup
import torch.nn as nn
from joblib import load


def eval(data_loader, state_in, len_input, d_model, len_output, model, scaler, criterion, device):
    pred_hist = []
    loss_hist = []
    
    model.eval()
    for i, (X,y) in enumerate(data_loader):
        i = i+1
        X = X.to(device)
        y = y.to(device)

        pred, _ = model.forward(X, state_in)
        pred_hist.append(pred.item())
        loss = criterion(pred.view(1), y.squeeze(0).squeeze(0)[0].unsqueeze(-1))
        loss_hist.append(loss.item())

        if i == len(data_loader)-(len_input+len_output):
            print('Prediction ended')
            break
    
#     pred_hist = scaler.inverse_transform(pred_hist)

    return pred_hist, loss_hist