import pandas as pd
import time
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load


def data_split(source, range = [0.6, 0.2, 0.2]):
    df = pd.read_csv(source).iloc[:,:6]
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit= 's')
    train_end = int(np.floor(len(df)*range[0]))
    valid_end = int(np.floor(len(df)*(range[0]+range[1])))
    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end : valid_end]
    test_df = df.iloc[valid_end:]
    train_df.to_csv('return-data/train-btc-raw.csv', index = False)
    valid_df.to_csv('return-data/valid-btc-raw.csv', index = False)
    test_df.to_csv('return-data/test-btc-raw.csv', index = False)

# encoding the timestamp data cyclically. See Medium Article.
def process_data(source, is_train = False):

    df = pd.read_csv(source)
        
    timestamps = [ts.split('+')[0] for ts in df['Open Time']]
    timestamps_hour = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour) for t in timestamps])
    timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

    #Calculate horly return
    df['Return'] = df['Close'].shift(1)/df['Close'] - 1

    df = df[['Return', 'Close', 'Volume', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']]
    df = df[1:]

    if is_train:
        #Scale Close and Volume data
        scaler = MinMaxScaler()
        df[['Return', 'Close', 'Volume']] = scaler.fit_transform(df[['Return', 'Close', 'Volume']])
        
        #save the scaler
        dump(scaler, 'train-return-scaler.joblib')

    if not is_train:
        try:
            scaler = load('train-return-scaler.joblib')
            df[['Return', 'Close', 'Volume']] = scaler.transform(df[['Return', 'Close', 'Volume']])
        except:
            print('Scaler file does not exist. Please process the train dataset first')
        
    return df

data_split('Data/btc-usdt-1hr.csv', range = [0.6, 0.2, 0.2])
train_dataset = process_data('return-data/train-btc-raw.csv', is_train= True)
valid_dataset = process_data('return-data/valid-btc-raw.csv')
test_dataset = process_data('return-data/test-btc-raw.csv')

train_dataset.to_csv('return-data/train-btc-dataset.csv', index=False)
valid_dataset.to_csv('return-data/valid-btc-dataset.csv', index=False)
test_dataset.to_csv('return-data/test-btc-dataset.csv', index=False)