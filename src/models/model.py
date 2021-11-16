import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials, SparkTrials, space_eval
import pickle


class Trainer(ABC):
    '''Basic training class, defines basic structures of training'''
    
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class XGBHyperoptTimeSeries(Trainer):
    '''Training XGBRegressor model for time series forecasting with Bayesian hyperparameter optimization'''

    def __init__(self):
        pass

    def fit(self, X, y, horizon, save_path='xgb.pkl'):
        '''Define basic training structure.

        Parameters
        ----------
        X : pandas data frame
            Data frame with predictor features.
        y : pandas series
            Pandas series with target data.
        horizon : int
            Size of forecasting horizon.

        Returns
        -------
        xgb.pkl : pickle file
            Trained model.
        '''

        def xgb_hyperopt(X, y, verbose=False, persistIterations=True):
            '''Bayesian hyperparameter optimization with Hyperopt

            Parameters
            ----------
            X : pandas data frame
                Data frame with predictor features.
            y : pandas series
                Pandas series with target data.

            Returns
            -------
            space_eval(parameters, best) : dict
                Best hyperparameters. 
            '''

            def objective(space):
                '''Function to minimize RMSE score

                Paramenters
                -----------
                space : python dict
                    Hyperparamenter space to check RMSE score.

                Returns
                -------
                score : float   
                    RMSE score.
                '''
                
                model = XGBRegressor(**space, seed=42, nthread=1)   

                score = np.sqrt(-cross_val_score(
                    model, 
                    X, 
                    y, 
                    cv=TimeSeriesSplit(n_splits=3, test_size=horizon), 
                    scoring='neg_mean_squared_error', 
                    verbose=False, 
                    n_jobs=-1
                    ).mean())

                return score

            parameters = {
                'max_depth' : hp.randint('max_depth', 50),
                'n_estimators': hp.randint('n_estimators', 1000),
                'learning_rate': hp.loguniform('learning_rate', -3, 1)
            }

            best = fmin(
                objective, 
                space=parameters, 
                algo=tpe.suggest, 
                max_evals=10, 
                trials=Trials(),
                rstate=np.random.RandomState(0)
            )

            return space_eval(parameters, best)

        best_xgb = xgb_hyperopt(X, y)
        xgb = XGBRegressor(**best_xgb, seed=42, nthread=-1)  
        xgb.fit(X, y)

        with open(save_path, 'wb') as f:
            pickle.dump(xgb, f)

    def predict(self, future, save_path='xgb.pkl'):
        '''Define basic prediction structure.

        Parameters
        ----------
        future : pandas data frame 
            Data frame with predictor features.

        Returns
        -------
        list(prediction) : list
            List with predictions.
        '''

        with open(save_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(future)

        return list(prediction)