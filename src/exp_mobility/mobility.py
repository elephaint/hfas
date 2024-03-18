#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
from hierts.reconciliation import apply_reconciliation_methods, hierarchy_cross_sectional, aggregate_bottom_up_forecasts, calc_level_method_error
from src.lib import prepare_HierarchicalLoss, HierarchicalLossObjective, HierarchicalLossMetric
from functools import partial
#%% Prepare data
# Data source:: https://www.kaggle.com/datasets/pronto/cycle-share-dataset?resource=download
# First 50794 lines are duplicate, these have already been removed
filepath_trips = CURRENT_PATH.joinpath("data/trip.csv")
filepath_weather = CURRENT_PATH.joinpath("data/weather.csv")
df_trips = pd.read_csv(filepath_trips)
df_weather = pd.read_csv(filepath_weather)
df_trips['date'] = pd.to_datetime(df_trips['starttime']).dt.date
df_weather['date'] = pd.to_datetime(df_weather['Date'])
df = df_trips.groupby(['date', 'from_station_id'])['tripduration'].count().reset_index()
df = df.rename(columns={'tripduration':'trip_count'})
df['from_station_id'] = df['from_station_id'].str.replace("Pronto shop 2", 'PS-2')
df['from_station_id'] = df['from_station_id'].str.replace("Pronto shop", 'PS-1')
df['from_station_id'] = df['from_station_id'].str.replace("8D OPS 02", 'OPS-1')
df['station_group'] = df['from_station_id'].map(lambda x: x.split('-')[0])
df['station'] = df['from_station_id'].map(lambda x: x.split('-')[1])
df['date'] = pd.to_datetime(df['date'])
df = df.merge(df_weather[['date', 'Mean_Temperature_F', 'Precipitation_In']], left_on='date', right_on='date', how='left')
#%% Create hierarchies
cross_sectional_aggregations = [['station_group'],
                                ['station']]
time_index = 'date'
target = 'trip_count'
bottom_timeseries = 'from_station_id'
start_train = df['date'].min()
end_train = pd.to_datetime("2016-05-31")
start_test = pd.to_datetime("2016-06-01")
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=bottom_timeseries)
#%% Create X and y
def create_forecast_set(df, df_Sc, time_index, target, bottom_timeseries, forecast_day=0):
    # Create target df
    if hasattr(df_Sc, "sparse"):
        Sc = df_Sc.sparse.to_coo().tocsc()
    else:
        Sc = df_Sc.values
    df_target = df.set_index([time_index, bottom_timeseries])[target].unstack(1, fill_value=0)
    targets = (Sc @ df_target.T.values).astype('float32')
    targets = pd.DataFrame(index=df_Sc.index, columns=df_target.T.columns, data=targets)
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
        rolling_target = group.rolling(window, min_periods=1).mean().astype('float32').droplevel([0, 1])
        rolling_target.name = f'{target}_lag{first_lag}_mavg{window}'
        X = pd.concat((X, rolling_target), axis=1)
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

    # Dropnans
    X = X.dropna()

    # Save and reset index, add aggregation and value columns as categoricals
    Xind = X.index
    X = X.reset_index(['Aggregation', 'Value'])
    X[['Aggregation', 'Value']] = X[['Aggregation', 'Value']].astype('category')
    # Return targets in the correct order
    targets = targets.loc[:, X.index.get_level_values(time_index).unique()]

    # Return forecast set
    return X, Xind, targets


X, Xind, targets = create_forecast_set(df, df_Sc, time_index, target, bottom_timeseries, forecast_day=0) 
#%% Train LightGBM model
# This requires LightGBM 3.x; 
params = {'seed': 0,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'verbosity': 1}
params['objective'] = None # LightGBM 3.x
# params['objective'] = fobj # LightGBM 4.x
# Create training set
Xb = X[X['Aggregation'] == bottom_timeseries]
y_train = Xb[target].loc[:end_train]
X_train = Xb.drop(columns=[target]).loc[:end_train]
train_set = lgb.Dataset(X_train, y_train)
# Create test set
y_test = Xb[target].loc[start_test:]
X_test = Xb.drop(columns=[target]).loc[start_test:]
test_dates = X_test.index.get_level_values(time_index).unique().sort_values()
# Prepare loss function for use in LightGBM
n_bottom_timeseries = df_Sc.shape[1]
n_bottom_timesteps = X_train.index.nunique()
hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                n_bottom_timesteps=n_bottom_timesteps, 
                                                df_Sc=df_Sc,
                                                df_St=None)
fobj = partial(HierarchicalLossObjective, 
               hessian=hessian, 
               n_bottom_timeseries=n_bottom_timeseries, 
               n_bottom_timesteps=n_bottom_timesteps, 
                Sc=Sc, 
                Scd=Scd)
# Train models: one with hierarchical loss, one with normal mse 
model_hmse = lgb.train(params, train_set, fobj=fobj)
params['objective'] = 'mse'
model_mse = lgb.train(params, train_set)
models = {'hmse': model_hmse, 'mse': model_mse}
X_test_ind = X_test[['Aggregation', 'Value']].reset_index().set_index(["Aggregation", "Value", time_index]).index
# Obtain forecasts for both models
forecasts = {}
for objective, model in models.items():
    # Make predictions on test set
    yhat_test = model.predict(X_test)
    # Create dataframe of predictions
    df_yhat_test = pd.Series(index = X_test_ind, data=yhat_test)
    df_yhat_test = df_yhat_test.droplevel(level="Aggregation", axis=0)
    forecasts_bottom_level = df_yhat_test.unstack([time_index]).loc[targets.loc[bottom_timeseries].index, test_dates]
    # Aggregate bottom-up forecasts to higher aggregations
    forecasts_model = aggregate_bottom_up_forecasts(forecasts_bottom_level, df_Sc, bottom_timeseries)
    forecasts_model = forecasts_model.astype('float64')
    forecasts[objective] = forecasts_model

forecasts_methods = pd.concat(forecasts)
forecasts_methods.index.names = ['Method', 'Aggregation', 'Value']

# Evaluate test set forecasts
test_targets = targets.loc[:, start_test:]
metrics = ['RMSE', 'MAE', 'SMAPE']
errors = pd.DataFrame()
for metric in metrics:
    error = calc_level_method_error(forecasts_methods=forecasts_methods, actuals=test_targets, metric=metric)
    error.columns = pd.MultiIndex.from_product([[metric], error.columns])
    errors = pd.concat((errors, error), axis=1)

errors    