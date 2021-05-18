import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot

import datetime

from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate, TimeSeriesSplit, GridSearchCV

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from joblib import dump, load


def date_series(s, dates, fill_val = np.NaN):
    """
    Assigns a time indexed sereis to a new time index date range
    INPUT:
        s (Sereis) - Pandas series with a DatetimeIndex
        dates (List[Datatime]) - Range of datetime values for new new index
        fill_val (Same type as items in s) - Default value to apply when new index hase dates not in original index
    OUPUT:
        Pandas series with date time index matching the input date values
    """
    return s.reindex(pd.DatetimeIndex(dates), fill_value = fill_val)


def code_feature_constructor(df, code, country, dates, max_lag = 3):
    """
    Constructs time series features used for supervised model for a single code and country
    INPUT:
        df (DataFrame) - Pandas dataframe containing Date, Quantity and StockCode columns
        code (String) - Stock codes selected for feature construction
        country (String) - Country selected for feature construction
        dates (List[Datetime]) - Range of dates for dataset index
        max_lag (Int) - Number of days to calculate lag values from differenced data 
    OUTPUT:
        Pandas dataframe with differencing and time series features for supervised model
    """
    part_df = pd.DataFrame()
    part_df['StockCode'] = date_series(pd.Series([code]), dates, code)
    part_df['Quantity'] = date_series(pd.Series(df['Quantity'].to_list(), pd.DatetimeIndex(df['Date'])), dates, 0)
    part_df['QuantityDiff'] = part_df['Quantity'].diff()
    part_df['UnitPrice'] = date_series(pd.Series(df['UnitPrice'].to_list(), pd.DatetimeIndex(df['Date'])), dates)
    part_df['DayOfWeek'] = part_df.index.dayofweek
    part_df['Month'] = part_df.index.month
    part_df['Country'] = date_series(pd.Series([country]), dates, country)
    
    # Add lag features based on input duration
    for lag in range(1, max_lag + 1, 1):
        part_df[f'Lag_{str(lag)}'] = part_df['QuantityDiff'].shift(lag)
    
    part_df.index = part_df.index.rename('Date')
    return part_df.sort_values('Date').ffill()


def feature_constructor(df, max_lag = 3):
    """
    Constructs time series features for supervised model for all codes and countries
    INPUT:
        df (DataFrame) - Pandas dataframe containing Date, Quantity and StockCode columns
        max_lag (Int) - Number of days to calculate lag values from differenced data 
    OUTPUT:
        new_df (DataFrame) - Pandas dataframe with time series fields for all records
    """
    dates = pd.date_range(df['Date'].min(), df['Date'].max())
    df['CodeCountry'] = df['StockCode'].str.cat(df['Country'],sep="|")
    code_country = df['CodeCountry'].drop_duplicates().to_numpy()
    args = ((df[df['CodeCountry'] == cc], cc.split('|')[0], cc.split('|')[1], dates, max_lag) for cc in code_country)
    df_list = list(map(lambda x: code_feature_constructor(*x), args))

    new_df = pd.concat(df_list).sort_values('Date')
    new_df['pad'] = 1
    return new_df


def format_data(df, max_lag = 7):
    """
    Master function to format all data for supervised training with time indexes and grouped by StockCode and Country
    INPUT:
        df (DataFrame) - Pandas dataframe containing Date, Quantity and StockCode columns
        max_lag (Int) - Number of days to calculate lag values from differenced data 
    OUTPUT:
        new_df (DataFrame) - Pandas dataframe with time series fields for all records
    """
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)
    df['Date'] = df['InvoiceDate'].dt.date
    df = df.groupby(['StockCode', 'Country', 'Date']).agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).sort_values('Date').reset_index()
    return feature_constructor(df, max_lag)


def train_test_data(df, codes, test_start, test_end):
    """
    Splits data into train and test sets based on specified test window and evaluation codes
    INPUT:
        df (DataFrame) - Pandas dataframe with data formatted for supervised learning
        codes (List[String]) - StockCodes targeted for model predictions
        test_start (Datetime) - Start time for testing window
        test_end (Datetime) - End time of testing window
    OUTPUT:
        train_X (DataFrame) - Pandas dataframe containing feature data for training
        train_y (Series) - Pandas series containing label data for training
        test_X (DataFrame) - Pandas dataframe containing feature data for testing
        test_y (Series) - Pandas series containing label data for testing
    """
    df = df.reset_index()
    test_df = df[df['StockCode'].isin(codes)][df['Date'] >= test_start][df['Date'] <= test_end].dropna()
    train_df = df[df['Date'] < test_start].dropna()
    test_X = test_df.drop(['Date', 'Quantity'], axis = 1)
    test_y = test_df['Quantity']
    train_X = train_df.drop(['Date', 'Quantity'], axis = 1)
    train_y = train_df['Quantity']
    return train_X, train_y, test_X, test_y


def experiment(X, y, prep, regrs, metrics):
    """
    Wrapper function to run experiment comparing a list of regression models
    INPUT:
        X (DataFrame) - Pandas dataframe containing feature data for training
        y (Series) - Pandas series containing label data for training
        prep (Pipeline) - Scikit-Learn pipeline for preprocessing feature data
        regrs (Model) - Scikit-Learn regression model
        metrics (List[String]) - list of metrics used for displaying experiment results
    OUTPUT:
        models (List[Pipeline Models]) - List of pipeline models evaluated in the experiment
        exp_df (DatFrame) - Pandas dataframe displaying the metric results of the experiment
    """
    exp_df_list = []
    models = []
    tscv = TimeSeriesSplit(n_splits=4)
    for model_name, model in regrs:
        if model_name in ['PoissonRegressor', 'HistGradientBoostingRegressor']:
            y = y.clip(lower = 0)
        pipe_model = make_pipeline(prep, model)
        cv_results = cross_validate(pipe_model, X, y, cv = tscv, scoring = metrics)
        metric_data = [[model_name] + [cv_results[f'test_{m}'].mean() for m in metrics]]
        tdf = pd.DataFrame(metric_data, columns = ['Model'] + metrics)
        exp_df_list.append(tdf)
        models.append(pipe_model)
    exp_df = pd.concat(exp_df_list)
    return models, exp_df

def get_metrics_df(model_name, y_true, y_pred):
    """
    Provides model evaluation metrics in easy to read format
    INPUT:
        model_name (Sting) - Name of the model for display purposes
        y_true (List,Array,Series) - True label values
        y_pred (List,Array,Series) - Predited label values
    OUTPUT:
        metric_df (DataFrame) - Pandas dataframe displaying metrics for input values
    """
    metric_df = pd.DataFrame({
        'Model': [model_name],
        'me': [max_error(y_true, y_pred)],
        'mae': [mean_absolute_error(y_true, y_pred)],
        'mse': [mean_squared_error(y_true, y_pred)],
        'mape': [mean_absolute_percentage_error(y_true, y_pred)],
        'r2': [r2_score(y_true, y_pred)]
    })
    return metric_df

