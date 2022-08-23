#%% Read packages
import pandas as pd
import numpy as np
from src.lib import calc_summing_matrix, apply_reconciliation_methods, aggregate_bottom_up_forecasts, hierarchical_obj_se, hierarchical_eval_se, create_levels, calc_level_method_rmse, hierarchical_obj_se_new, hierarchical_eval_se_new
import matplotlib.pyplot as plt
from typing import List, Tuple
import lightgbm as lgb
from numba.typed import List
#%% Fit forecasting model to a set of time series and their aggregations
# Read data
df = pd.read_csv('src/exp_tourism/tourism.csv', index_col=0)
df['Quarter'] = pd.PeriodIndex(df['Quarter'].str[0:4] + '-' + df['Quarter'].str[5:], freq='q')
# Set aggregations and target
aggregation_cols = ['State', 'Region', 'Purpose']
aggregations = [['State'],
                ['State', 'Region'],
                ['State', 'Purpose'],
                ['Purpose']]
# aggregations = [['State']]
target = 'Trips'
time_index = 'Quarter'
end_train = '2015Q4'
start_test = '2016Q1'
# Calculate summing matrix
df_S = calc_summing_matrix(df, aggregation_cols, aggregations)
# Create a forecasting model for each time series in the aggregation matrix df_S
# Add bottom time series names to original df
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
df['bottom_timeseries'] = pd.Categorical(df['bottom_timeseries'], df_S.columns)
df_target = df.set_index(['bottom_timeseries', time_index])[target].unstack(0)
#%% Create single LGB model for bottom-level and aggregate forecasts
# Preprocessing for forecast
def create_forecast_set(data, forecast_day, time_index):
    lags = np.arange(1, 8)
    group = data.groupby(['Aggregation', 'Value'])
    X = pd.DataFrame(index=data.index)
    for lag in lags:
        X['target_lag'+str(lag)] = group.shift(lag + forecast_day)
    
    X['weekday'] = X.index.get_level_values(time_index).weekday
    X = X.dropna()

    return X

actuals = df_S @ df_target.T
df_target_flat = actuals.stack([time_index]).sort_index(level=[time_index, 'Aggregation', 'Value'])
df_target_flat.name = target
X_lag = create_forecast_set(df_target_flat, forecast_day=0, time_index=time_index)
X = pd.concat((df_target_flat, X_lag), axis=1, join='inner')
Xind = X.index
X = X.reset_index(['Aggregation', 'Value'])
X[['Aggregation', 'Value']] = X[['Aggregation', 'Value']].astype('category')
#%% Global-agg model
y_train = X[target].loc[:end_train]
X_train = X.drop(columns=[target]).loc[:end_train]
# Create levels for training
levels_train = create_levels(X_train.loc[:end_train].reset_index()[[time_index, 'Value']])
level_weights = List([1, 1, 1])
train_set = lgb.Dataset(X_train, y_train)
eval_set = lgb.Dataset(X_train, y_train)
train_set.levels = levels_train
train_set.level_weights = level_weights
eval_set.levels = levels_train
eval_set.level_weights = level_weights
params = lgb.LGBMRegressor().get_params()
params['seed'] = 0
params['random_hierarchy'] = False
params['metric'] = None
params['n_estimators'] = 100
# model = lgb.train(params, train_set, valid_sets=eval_set, fobj=hierarchical_obj_se, feval=hierarchical_eval_se)
model = lgb.train(params, train_set)
# Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
yhat = model.predict(X.drop(columns=[target]))
df_yhat = pd.Series(index=Xind, data=yhat)
forecasts = df_yhat.unstack([time_index]).loc[actuals.index, X.index.get_level_values(time_index).unique()]
actuals = actuals.loc[:, X.index.get_level_values(time_index).unique()]
residuals = (forecasts - actuals)
#%% Global-bottom-up model - v1
Xb_ind = pd.DataFrame(index=Xind).loc['Bottom level'].index
Xb = X[X['Aggregation'] == 'Bottom level']
y_train = Xb[target].loc[:end_train]
X_train = Xb.drop(columns=[target]).loc[:end_train]
# Create levels for training
levels_train = create_levels(X_train.loc[:end_train].reset_index()[[time_index, 'Value']])
level_weights = List([1, 1, 1])
train_set = lgb.Dataset(X_train, y_train)
eval_set = lgb.Dataset(X_train, y_train)
train_set.levels = levels_train
train_set.level_weights = level_weights
eval_set.levels = levels_train
eval_set.level_weights = level_weights
params = lgb.LGBMRegressor().get_params()
params['seed'] = 0
params['random_hierarchy'] = False
params['metric'] = None
params['n_estimators'] = 100
model = lgb.train(params, train_set, valid_sets=eval_set, fobj=hierarchical_obj_se, feval=hierarchical_eval_se)
# model = lgb.train(params, train_set)
# Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
yhat = model.predict(Xb.drop(columns=[target]))
df_yhat = pd.Series(index=Xb_ind, data=yhat)
forecasts_bu = df_yhat.unstack([time_index]).loc[actuals.loc['Bottom level'].index, Xb.index.get_level_values(time_index).unique()]
#%% Global-bottom-up model - v2
Xb_ind = pd.DataFrame(index=Xind).loc['Bottom level'].index
Xb = X[X['Aggregation'] == 'Bottom level']
y_train = Xb[target].loc[:end_train]
X_train = Xb.drop(columns=[target]).loc[:end_train]
# Create levels for training
train_set = lgb.Dataset(X_train, y_train)
eval_set = lgb.Dataset(X_train, y_train)
# Add attributes for training
train_set.S = df_S.values
train_set.y = actuals.loc[:, :end_train].values
train_set.n_levels = Xind.get_level_values('Aggregation').nunique() 
eval_set.S = train_set.S
eval_set.y = train_set.y
eval_set.n_levels = train_set.n_levels
params = lgb.LGBMRegressor().get_params()
params['seed'] = 0
params['random_hierarchy'] = False
params['metric'] = None
params['n_estimators'] = 100
model = lgb.train(params, train_set, valid_sets=eval_set, fobj=hierarchical_obj_se_new, feval=hierarchical_eval_se_new)
# model = lgb.train(params, train_set)
# Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
yhat = model.predict(Xb.drop(columns=[target]))
df_yhat = pd.Series(index=Xb_ind, data=yhat)
forecasts_bu = df_yhat.unstack([time_index]).loc[actuals.loc['Bottom level'].index, Xb.index.get_level_values(time_index).unique()]

#%% Reconciliation
# All forecasts
residuals_train = residuals.loc[:, :end_train]
forecasts_test = forecasts.loc[:, start_test:]
forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, residuals_train)
# Bottom-up forecasts
forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu, df_S)
residuals_bu = (forecasts_bu - actuals)
forecasts_bu_test = forecasts_bu.loc[:, start_test:]
forecasts_method = pd.concat({'bottom-up': forecasts_bu_test}, names=['Method'])
forecasts_methods = pd.concat((forecasts_method, forecasts_methods), axis=0)
# Calculate error for all levels and methods
rmse, rel_rmse = calc_level_method_rmse(forecasts_methods, actuals, base='bottom-up')