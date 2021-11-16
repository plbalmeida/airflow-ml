import numpy as np
import pandas as pd
import os


def make_initial_dataset(feature_name, start, periods, freq):
    '''Generates first time series.

    Parameters
    ----------
    feature_name : str
        Feature name for time series.
    start : str
        String in the mm/dd/yyyy format
    periods : int
        Number of periods to generate.
    freq : str
        Frequency strings can have multiples, e.g. ‘5H’. 

    Returns
    -------
    df : pandas dataframe
        First time series data.
    '''
    
    timestamp = pd.date_range(start=start, periods=periods, freq=freq)
    df = pd.DataFrame(timestamp, columns=['timestamp'])
    df[feature_name] = np.random.randint(0, 100, size=(len(timestamp)))
    
    return df


def make_incremental_dataset(df, feature_name, incremental_periods, freq):
    '''Generates incremental time series.

    Parameters
    ----------
    df : pandas dataframe
        Data frame on raw data directory.
    feature_name : str
        Feature name for time series.
    incremental_periods : int
        Number of periods to generate.
    freq : str
        Frequency strings can have multiples, e.g. ‘5H’.

    Returns
    -------
    df : pandas dataframe
        Time series data with incremental data.
    '''

    timestamp = pd.date_range(start=df.timestamp.max() + pd.Timedelta('1D'), periods=incremental_periods, freq=freq)
    new_df = pd.DataFrame(timestamp, columns=['timestamp'])
    new_df[feature_name] = np.random.randint(0, 100, size=(len(timestamp)))
    df = pd.concat([df, new_df])

    return df


if __name__ == '__main__':

    path = os.getcwd() + '/data/raw/'

    # run once time
    if len(os.listdir(os.getcwd() + '/data/raw')) == 0:
        vendas = make_initial_dataset(feature_name='vendas', start='1/1/2018', periods=365, freq='D')
        vendas.to_csv(path + 'vendas.csv', index=False)

    # run always
    vendas = pd.read_csv(path + 'vendas.csv')
    vendas['timestamp'] = pd.to_datetime(vendas['timestamp'])
    vendas = make_incremental_dataset(df=vendas, feature_name='vendas', incremental_periods=5, freq='D')
    vendas.to_csv(path + 'vendas.csv', index=False)