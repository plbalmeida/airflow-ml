import pandas as pd
import os
import pickle
from model import XGBHyperoptTimeSeries


if __name__ == '__main__':
    vendas = pd.read_csv('/home/ana/Documentos/airflowML/data/processed/vendas.csv')
    vendas['timestamp'] = pd.to_datetime(vendas['timestamp'])
    horizon = 5
    future = vendas.iloc[-horizon:].drop(['timestamp', 'vendas'], axis=1)
    timestamp = pd.date_range(start=vendas.timestamp.max() + pd.Timedelta('1D'), periods=horizon, freq='D')
    future['timestamp'] = timestamp
    future.to_csv('/home/ana/Documentos/airflowML/data/processed/future.csv', index=False)
    features_to_shift = vendas.drop(['timestamp', 'vendas'], axis=1).columns
    vendas[features_to_shift] = vendas[features_to_shift].shift(horizon)
    vendas = vendas.dropna() 
    X = vendas.drop(['timestamp', 'vendas'], axis=1)
    y = vendas.vendas
    model = XGBHyperoptTimeSeries()
    model.fit(X, y, horizon=horizon, save_path='/home/ana/Documentos/airflowML/models/xgb.pkl')