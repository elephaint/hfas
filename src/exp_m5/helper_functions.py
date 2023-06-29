import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, issparse
from pathlib import Path
#%% Read data
def read_m5(first_date='2012-01-01', last_date='2016-05-22', store_level=True, store_id=0):
    directory = Path(__file__).parent
    filename = directory.joinpath('data/m5_dataset_products.parquet')
    df = pd.read_parquet(filename, 
                        columns = ['sales', 'date', 'state_id_enc', 'store_id_enc', 'cat_id_enc', 
                                    'dept_id_enc', 'item_id_enc', 'snap_CA', 'snap_TX', 'snap_WI',
                                    'event_type_1_enc', 'event_type_2_enc', 'weeks_on_sale', 'sell_price'])
    if store_level:
        df = df[(df['date'] <= last_date) & (df['date'] >= first_date) 
            & (df['weeks_on_sale'] > 0) & (df['store_id_enc'] == store_id)]
    else:
        df = df[(df['date'] <= last_date) & (df['date'] >= first_date) 
            & (df['weeks_on_sale'] > 0)]

    df = df.sort_values(by=['store_id_enc', 'item_id_enc', 'date']).reset_index(drop=True)

    return df
#%% Feature engineering
def create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day):
    # Add bottom-level time series to dataframe
    bottom_timeseries = pd.DataFrame(index=df_S.columns.str.split(pat='-',expand=True).set_names(aggregation_cols)).reset_index()
    bottom_timeseries[aggregation_cols] = bottom_timeseries[aggregation_cols].astype('int')
    bottom_timeseries['bottom_timeseries'] = bottom_timeseries[aggregation_cols].astype(str).agg('-'.join, axis=1)
    bottom_timeseries['bottom_timeseries'] = pd.Categorical(bottom_timeseries['bottom_timeseries'], df_S.columns)
    df[aggregation_cols] = df[aggregation_cols].astype('int')
    df = df.merge(bottom_timeseries, how='left', left_on=aggregation_cols, right_on=aggregation_cols)
    # Create target df
    if hasattr(df_S, "sparse"):
        print("S is sparse")
        S = csc_matrix(df_S.sparse.to_coo())
    else:
        print("S is dense")
        S = df_S.values
    df_target = df.set_index([time_index, 'bottom_timeseries'])[target].unstack(1, fill_value=0)
    targets = (S @ df_target.T.values).astype('float32')
    targets = pd.DataFrame(index=df_S.index, columns=df_target.T.columns, data=targets)
    targets_flat = targets.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
    targets_flat.name = target    
    # Create lags
    lags = np.concatenate((np.arange(1, 8), [28, 56, 364]))
    group = targets_flat.groupby(['Aggregation', 'Value'])
    # Create X, and set target
    X = pd.DataFrame(index=targets_flat.index)
    target = targets_flat.name
    X[target] = targets_flat.astype('float32')
    # Create lags
    for lag in lags:
        X[f'{target}_lag{lag}'] = group.shift(lag + forecast_day).astype('float32')
    # Create moving average lags
    windows = [7, 28, 56]
    first_lag = 1 + forecast_day
    group = X.groupby(['Aggregation', 'Value'])[f'{target}_lag{first_lag}']
    for window in windows:
        X[f'{target}_lag{first_lag}_mavg{window}'] = group.transform(lambda x: x.rolling(window, min_periods=1).mean()).astype('float32')
    # Add weekday and target
    level_values = X.index.get_level_values(time_index)
    X['dayofweek'] = (level_values.isocalendar().day).values
    X['dayofmonth'] = (level_values.day).values
    X['weekofyear'] = (level_values.isocalendar().week).values
    X['monthofyear'] = (level_values.month).values
    X['dayofweek'] = X['dayofweek'].astype('category')
    X['dayofmonth'] = X['dayofmonth'].astype('category')
    X['weekofyear'] = X['weekofyear'].astype('category')
    X['monthofyear'] = X['monthofyear'].astype('category')
    # Prices (use average price for aggregations)
    df_prices = df.set_index([time_index, 'bottom_timeseries'])['sell_price'].unstack(1)
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
    prices = S @ df_prices.T.values
    prices = pd.DataFrame(index=df_S.index, columns=df_prices.T.columns, data=prices)
    S_sum = np.array(S.sum(axis=1)).squeeze()[:, None]
    prices = prices / S_sum
    df_prices_flat = prices.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
    df_price_change_flat = df_prices_flat.groupby(['Aggregation', 'Value']).shift(1) / df_prices_flat.groupby(['Aggregation', 'Value']).shift(0) - 1
    X['sell_price_avg'] = df_prices_flat.astype('float32')
    X['sell_price_change'] = df_price_change_flat.astype('float32')
    # Weeks on sale (use average for aggregations)
    df_weeks_on_sale = df.set_index([time_index, 'bottom_timeseries'])['weeks_on_sale'].unstack(1, fill_value=0)
    weeks_on_sale = S @ df_weeks_on_sale.T.values
    weeks_on_sale = pd.DataFrame(index=df_S.index, columns=df_weeks_on_sale.T.columns, data=weeks_on_sale)
    weeks_on_sale = weeks_on_sale / S_sum
    df_weeks_on_sale_flat = weeks_on_sale.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
    X['weeks_on_sale_avg'] = df_weeks_on_sale_flat.astype('float32')
    # Add other information
    other_cols = ['snap_CA', 'snap_TX', 'snap_WI', 'event_type_1_enc', 'event_type_2_enc']
    df_other_cols = df[['date'] + other_cols]
    df_other_cols = df_other_cols.set_index('date')
    df_other_cols[other_cols] = df_other_cols[other_cols].astype('category')
    df_other_cols = df_other_cols[~df_other_cols.index.duplicated()]
    X = X.join(df_other_cols, how='left')
    # Retain last two years as training data
    X = X.swaplevel(0, -1).loc['2013-01-01':].swaplevel(0, -1)
    # Dropnans
    X = X.dropna()
    # Save and reset index, add aggregation and value columns as categoricals
    Xind = X.index
    X = X.reset_index(['Aggregation', 'Value'])
    X[['Aggregation', 'Value']] = X[['Aggregation', 'Value']].astype('category')
    # Return targets in the correct order
    targets = targets.loc[:, X.index.get_level_values(time_index).unique()]

    return X, Xind, targets