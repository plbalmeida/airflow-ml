import pandas as pd
import os

def get_lag_features(df, features_to_lag, number_of_lags):
    '''Creates lag features from the requested features
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all features.
    features_to_lag : list 
        List of features that will be made the lags features.
    number_of_lags: int
        Maximum number of lags per feature.
    
    Returns
    -------
    df: pandas dataframe
        Pandas data frame with lag features.
    '''
    
    for feature in df[features_to_lag].columns: 
        shifts = range(1, number_of_lags + 1)
        shifted_data = pd.DataFrame({'{}_lag{}'.format(str(feature), str(lag)) : df[feature].shift(lag) for lag in shifts})
        df = df.merge(shifted_data, how='left', left_index=True, right_index=True)
        df = df.fillna(0)
        
    return df


if __name__ == '__main__':
    vendas = pd.read_csv('/home/ana/Documentos/airflowML/data/raw/vendas.csv')
    vendas = get_lag_features(df=vendas, features_to_lag=['vendas'], number_of_lags=15)
    vendas.to_csv('/home/ana/Documentos/airflowML/data/processed/vendas.csv', index=False)