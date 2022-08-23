#%% Read packages
import pandas as pd
import numpy as np
from src.lib import calc_summing_matrix, apply_all_reconciliation_methods, aggregate_bottom_up_forecasts, hierarchical_obj_se, hierarchical_eval_se, create_levels, calc_level_method_rmse
import matplotlib.pyplot as plt
from typing import List, Tuple
from hts.utilities.load_data import load_mobility_data
import lightgbm as lgb
from numba.typed import List
#%% Fit forecasting model to a set of time series and their aggregations
hd = load_mobility_data()
df = hd.drop(columns = ['total', 'CH', 'SLU', 'BT', 'OTHER', 'temp', 'precipitation'])
df = df.stack().reset_index()
df.rename(columns = {'level_1':'item', 0:'target'}, inplace=True)
df['group'] = df['item'].str.split('-', 1, expand=True)[0].astype("category")
df['item'] = df['item'].str.split('-', 1, expand=True)[1].astype("category")
df = df[['starttime', 'group', 'item', 'target']]
df = df.merge(hd.reset_index()[['starttime', 'temp', 'precipitation']], how='left', left_on=['starttime'], right_on=['starttime'])
df.rename(columns = {'starttime':'date'}, inplace=True)
#%% Reconciliation model: Set aggregations and target
aggregation_cols = ['group', 'item']
aggregations = [['group']]
target = 'target'
exogenous = ['temp', 'precipitation']
# Calculate summing matrix
df_S = calc_summing_matrix(df, aggregation_cols, aggregations)
# Create a forecasting model for each time series in the aggregation matrix df_S
# Add bottom time series names to original df
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
df['bottom_timeseries'] = pd.Categorical(df['bottom_timeseries'], df_S.columns)
df_target = df.set_index(['bottom_timeseries', 'date'])[target].unstack(0)
df_exogenous = df.set_index(['bottom_timeseries', 'date'])[exogenous].unstack(0)
temp = df_exogenous['temp'].mean(1)
#%% Create single LGB model for bottom-level and aggregate forecasts
# Preprocessing for forecast
def create_forecast_set(data, forecast_day):
    lags = np.arange(1, 8)
    group = data.groupby(['Aggregation', 'Value'])
    X = pd.DataFrame(index=data.index)
    for lag in lags:
        X['target_lag'+str(lag)] = group.shift(lag + forecast_day)
    
    X['weekday'] = X.index.get_level_values('date').weekday
    X = X.dropna()

    return X
actuals = df_S @ df_target.T
df_target_flat = actuals.stack(['date']).sort_index()
df_target_flat.name = target
df_precipitation_flat = (df_S @ df_exogenous['precipitation'].T).stack(['date']).loc[df_target_flat.index]
df_precipitation_flat.name = 'precipitation'
X_lag = create_forecast_set(df_target_flat, 0)
X = pd.concat((df_target_flat, X_lag, df_precipitation_flat), axis=1, join='inner')
Xind = X.index
X = X.reset_index(['Aggregation', 'Value'])
X[['Aggregation', 'Value']] = X[['Aggregation', 'Value']].astype('category')
#%% Global-agg model
y_train = X['target'].loc[:'2015-12-31']
X_train = X.drop(columns=['target']).loc[:'2015-12-31']
# Create levels for training
levels_train = create_levels(X_train.loc[:'2015-12-31'].reset_index()[['date', 'Value']])
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
yhat = model.predict(X.drop(columns=['target']))
df_yhat = pd.Series(index=Xind, data=yhat)
forecasts = df_yhat.unstack(['date']).loc[actuals.index, X.index.get_level_values('date').unique()]
actuals = actuals.loc[:, X.index.get_level_values('date').unique()]
residuals = (forecasts - actuals)
#%% Global-bottom-up model
Xb_ind = pd.DataFrame(index=Xind).loc['Bottom level'].index
Xb = X[X['Aggregation'] == 'Bottom level']
y_train = Xb['target'].loc[:'2015-12-31']
X_train = Xb.drop(columns=['target']).loc[:'2015-12-31']
# Create levels for training
levels_train = create_levels(X_train.loc[:'2015-12-31'].reset_index()[['date', 'Value']])
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
yhat = model.predict(Xb.drop(columns=['target']))
df_yhat = pd.Series(index=Xb_ind, data=yhat)
forecasts_bu = df_yhat.unstack(['date']).loc[actuals.loc['Bottom level'].index, Xb.index.get_level_values('date').unique()]
#%% Reconciliation
# All forecasts
residuals_train = residuals.loc[:, :'2015-12-31']
forecasts_test = forecasts.loc[:, '2016-01-01':]
forecasts_methods = apply_all_reconciliation_methods(forecasts_test, df_S, residuals_train)
# Bottom-up forecasts
forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu, df_S)
residuals_bu = (forecasts_bu - actuals)
forecasts_bu_test = forecasts_bu.loc[:, '2016-01-01':]
forecasts_method = pd.concat({'bottom-up': forecasts_bu_test}, names=['Method'])
forecasts_methods = pd.concat((forecasts_method, forecasts_methods), axis=0)
# Calculate error for all levels and methods
rmse, rel_rmse = calc_level_method_rmse(forecasts_methods, actuals, base='bottom-up')