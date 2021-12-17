import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class Dataset(Dataset):

    def __init__(self, path, len_input):
        self.df = pd.read_csv(path)
        self.len_input = len_input

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # self.df = self.df[['Close', 'Volume', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']]
        start = idx
        end = idx + self.len_input

        X = self.df[start: end].values
        y = np.array(self.df.iloc[end,:])

        X = torch.DoubleTensor(X)
        y = torch.DoubleTensor(y).unsqueeze(0)

        return X ,y