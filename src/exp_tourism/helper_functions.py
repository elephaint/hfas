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
    df['Quarter'] = pd.PeriodIndex(df['Quarter'].str[0:4] + '-' + df['Quarter'].str[5:], freq='q').astype("datetime64[ns]")

    return df
#%% Create forecast set
def create_forecast_set(df, df_Sc, aggregation_cols, time_index, target, forecast_quarter=0):

    # Add bottom-level time series to dataframe
    bottom_timeseries = pd.DataFrame(index=df_Sc.columns.str.split(pat='-',expand=True).set_names(aggregation_cols)).reset_index()
    bottom_timeseries['bottom_timeseries'] = bottom_timeseries[aggregation_cols].astype(str).agg('-'.join, axis=1)
    bottom_timeseries['bottom_timeseries'] = pd.Categorical(bottom_timeseries['bottom_timeseries'], df_Sc.columns)
    df = df.merge(bottom_timeseries, how='left', left_on=aggregation_cols, right_on=aggregation_cols)
    # Create target df
    if hasattr(df_Sc, "sparse"):
        print("S is sparse")
        S = df_Sc.sparse.to_coo().tocsc()
    else:
        print("S is dense")
        S = df_Sc.values
    df_target = df.set_index([time_index, 'bottom_timeseries'])[target].unstack(1, fill_value=0)
    targets = (S @ df_target.T.values).astype('float32')
    targets = pd.DataFrame(index=df_Sc.index, columns=df_target.T.columns, data=targets)
    targets_flat = targets.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
    targets_flat.name = target    
    # Create lags
    lags = np.arange(1, 12)
    group = targets_flat.groupby(['Aggregation', 'Value'])
    # Create X, and set target
    X = pd.DataFrame(index=targets_flat.index)
    target = targets_flat.name
    X[target] = targets_flat.astype('float32')
    # Create lags
    for lag in lags:
        X[f'{target}_lag{lag}'] = group.shift(lag + forecast_quarter).astype('float32')
    # Create moving average lags
    windows = [8]
    first_lag = 1 + forecast_quarter
    group = X.groupby(['Aggregation', 'Value'])[f'{target}_lag{first_lag}']
    for window in windows:
        X[f'{target}_lag{first_lag}_mavg{window}'] = group.transform(lambda x: x.rolling(window, min_periods=1).mean()).astype('float32')
    # Add year
    level_values = X.index.get_level_values(time_index)
    X["year"] = (level_values.year).values
    X["year"] = X["year"].astype('category')
    # Dropnans
    X = X.dropna()
    # Save and reset index, add aggregation and value columns as categoricals
    Xind = X.index
    X = X.reset_index(['Aggregation', 'Value'])
    X[['Aggregation', 'Value']] = X[['Aggregation', 'Value']].astype('category')
    # Return targets in the correct order
    targets = targets.loc[:, X.index.get_level_values(time_index).unique()]

    return X, Xind, targets