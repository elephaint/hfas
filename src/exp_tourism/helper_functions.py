import pandas as pd
import numpy as np
from pathlib import Path
#%% M5 aggregations
def get_aggregations():
    cross_sectional_aggregations = [['State'],
                                    ['State', 'Region'],
                                    ['State', 'Purpose'],
                                    ['Purpose']]

    temporal_aggregations = [['year']]

    return cross_sectional_aggregations, temporal_aggregations
#%% Read data
def read_tourism():
    directory = Path(__file__).parent
    filename = directory.joinpath('data/tourism.csv')

    df = pd.read_csv(str(filename), index_col=0)
    df['Quarter'] = pd.PeriodIndex(df['Quarter'].str[0:4] + '-' + df['Quarter'].str[5:], freq='q')

    return df
#%% Create forecast set
def create_forecast_set(df, df_Sc, aggregation_cols, time_index, target, forecast_day=0):

    actuals = df_S @ df_target.T
    df_target_flat = actuals.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
    df_target_flat.name = target

    lags = np.arange(1, 8)
    group = data.groupby(['Aggregation', 'Value'])
    X = pd.DataFrame(index=data.index)
    for lag in lags:
        X['target_lag'+str(lag)] = group.shift(lag + forecast_day)
    
    X['weekday'] = X.index.get_level_values(time_index).weekday
    X = X.dropna()

    return X, Xind, targets