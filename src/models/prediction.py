import pandas as pd
import os
import pickle
from model import XGBHyperoptTimeSeries


if __name__ == '__main__':

    path = os.getcwd()
    future = pd.read_csv(path + '/data/processed/future.csv')
    model = XGBHyperoptTimeSeries()
    pred = model.predict(future=future.drop('timestamp', axis=1), save_path=path + '/models/xgb.pkl')
    forecast = pd.DataFrame({'forecast': pred})
    forecast['timestamp'] = future.timestamp
    forecast = forecast[['timestamp', 'forecast']]
    now = str(pd.Timestamp.now())
    now = now[:19].replace('-', '').replace(' ', '').replace(':', '')
    forecast.to_csv(path + '/data/output/forecast_{}.csv'.format(now), index=False)