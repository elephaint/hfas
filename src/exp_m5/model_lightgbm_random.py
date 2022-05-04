# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:01:40 2022

@author: ospra
"""
#%% Import packages
import numpy as np
import pandas as pd
import polars as pl
import time
import shutil
from numba import njit, prange
from numba.typed import List, Dict
from numba.core import types
import lightgbm as lgb
#%% Read data
data = pl.scan_parquet('datasets/m5/m5_dataset_products.parquet')
subset = data.filter((pl.col('date') <=  pl.date(2016, 5, 22)) & 
                     (pl.col('date') >=  pl.date(2014, 1, 1)) &
                     (pl.col('weeks_on_sale') > 0) &
                     (pl.col('store_id_enc') == 0))\
             .sort(by=['store_id_enc','item_id_enc','date'])\
             .collect()\
             .to_pandas()
#%% Preprocessing for forecast
cols_unknown = ['sales_lag1', 'sales_lag2',
   'sales_lag3', 'sales_lag4', 'sales_lag5', 'sales_lag6', 'sales_lag7',
   'sales_lag1_mavg7', 'sales_lag1_mavg28', 'sales_lag1_mavg56',
   'sales_lag7_mavg7', 'sales_lag7_mavg28', 'sales_lag7_mavg56',
   'sales_short_trend', 'sales_long_trend', 'sales_year_trend',
   'sales_item_long_trend', 'sales_item_year_trend']

cols_known = ['date','item_id_enc', 'dept_id_enc', 'cat_id_enc',
   'snap_CA', 'snap_TX', 'snap_WI', 'event_type_1_enc',
   'event_type_2_enc', 'sell_price',
   'sell_price_change', 'sell_price_norm_item', 'sell_price_norm_dept',
   'weeks_on_sale', 'dayofweek', 'dayofmonth',
   'weekofyear', 'monthofyear',
   'sales_lag364', 'sales_lag28_mavg7',
   'sales_lag28_mavg28', 'sales_lag28_mavg56', 'sales_lywow_trend',
   'sales_lag28', 'sales_lag56']

def create_forecastset(data, cols_unknown, cols_known, forecast_day):
    X_unknown = data.groupby(['store_id_enc','item_id_enc'])[cols_unknown].shift(forecast_day)
    X_known = data[cols_known]
    X = pd.concat((X_known, X_unknown), axis=1)
    y = data[['date','sales']]
    
    return X, y
#%% Create hierarchical squared error with daily aggregation
@njit(fastmath = True, parallel=True)
def hierarchical_loss(yhat, y, levels, level_weights, feval, params):
    dates = levels[0].T
    n_levels = len(levels) + 1
    n_dates = dates.shape[0]
    level_cnt = np.ones_like(y)
    loss = (level_weights[-1] / n_levels) * feval(yhat, y, params, level_cnt)
    for i in prange(n_dates):
        date = dates[i]
        yhatd = yhat[date]
        yd = y[date]
        # Level loss per day
        for j, level in enumerate(levels[1:]):
            leveld = level[date] * 1.
            yhatd_level = yhatd @ leveld
            yd_level = yd @ leveld
            level_cnt = np.maximum(leveld.sum(0), 1)
            loss += (level_weights[j + 1] / n_levels) * feval(yhatd_level, yd_level, params, level_cnt)
        # Daily loss - ugly way of getting Numba to accept the sum of yd and yhatd
        syhatd = np.zeros((1, 1))
        syd = np.zeros((1, 1))
        syhatd[0] = np.sum(yhatd) 
        syd[0] = np.sum(yd)
        level_cnt = np.maximum(len(yd), 1)
        loss += (level_weights[0] / n_levels) * feval(syhatd, syd, params, level_cnt)
        
    return loss  

@njit(fastmath = True, parallel=True)
def hierarchical_gradhess(yhat, y, levels, level_weights, fobj, params):
    dates = levels[0].T
    n_levels = len(levels)
    n_dates = dates.shape[0]
    level_cnt = np.ones_like(y)
    gradient, hessian = fobj(yhat, y, params, level_cnt)
    gradient, hessian = (level_weights[-1] / n_levels) * gradient, (level_weights[-1] / n_levels) * hessian
    # gradient, hessian = np.zeros_like(y), np.zeros_like(y)
    for i in prange(n_dates):
        date = dates[i]
        yhatd = yhat[date]
        yd = y[date]
        # Level loss gradient per day
        for j, level in enumerate(levels[1:]):
            leveld = level[date] * 1.
            yhatd_leveld = yhatd @ leveld
            yd_leveld = yd @ leveld
            level_cnt = np.maximum(leveld.sum(0), 1)
            gradientd_leveld, hessiand_leveld = fobj(yhatd_leveld, yd_leveld, params, level_cnt)
            gradient[date] += (level_weights[j + 1] / n_levels) * (leveld @ gradientd_leveld)
            hessian[date] += (level_weights[j + 1] / n_levels) * (leveld @ hessiand_leveld)
        # Daily gradient loss
        level_cnt = np.maximum(len(yd), 1)
        gradientd, hessiand = fobj(np.sum(yhatd), np.sum(yd), params, level_cnt)        
        gradient[date] += (level_weights[0] / n_levels) * gradientd
        hessian[date] += (level_weights[0] / n_levels) * hessiand
           
    return gradient, hessian

# squared error gradient and hessian
@njit(fastmath = True)
def fobj_se(yhat, y, params, level_cnt):
    gradient = (yhat - y) / level_cnt 
    hessian = np.ones_like(y)

    return gradient, hessian

# squared error metric
@njit(fastmath = True)
def feval_se(yhat, y, params, level_cnt):
    loss = 0.5 * np.square(yhat - y)
    loss /= level_cnt
    
    return np.sum(loss)

# Lightgbm objective function wrapper
def hierarchical_obj_se(preds, train_data):
    assert hasattr(train_data, 'levels'), 'Train data should contain dictionary with level information for hierarchical loss'
    assert hasattr(train_data, 'params'), 'Train data should contain parameter dict'   
    params = Dict().empty(
                    key_type=types.unicode_type,
                    value_type=types.float64[:])
    level_dates = train_data.levels[0]
    levels, level_weights = create_random_levels(level_dates, seed = 0)
    y = train_data.get_label().astype(np.float64)
    gradient, hessian = hierarchical_gradhess(preds, y, levels, level_weights, fobj_se, params)
    
    return gradient, hessian

# Lightgbm metric function wrapper
def hierarchical_eval_se(preds, eval_data):
    assert hasattr(eval_data, 'levels'), 'Eval data should contain dictionary with level information for hierarchical loss'   
    assert hasattr(eval_data, 'params'), 'Eval data should contain parameter dict'   
    params = Dict().empty(
                    key_type=types.unicode_type,
                    value_type=types.float64[:])
    # levels = eval_data.levels
    # level_weights = eval_data.level_weights
    level_dates = eval_data.levels[0]
    levels, level_weights = create_random_levels(level_dates, seed = eval_data.params['seed'])
    y = eval_data.get_label().astype(np.float64)
    loss = hierarchical_loss(preds, y, levels, level_weights, feval_se, params)
        
    return 'hierarchical_se', loss / len(y) , False

# Function to create levels for categorical columns
def create_levels(df):
    levels = List()
    cols = df.columns
    for col in cols:
        level = pd.get_dummies(df[col]).to_numpy().astype(bool)
        levels.append(level)
    
    return levels

# Function to create random levels for categorical columns
def create_random_levels(level_dates, n_categories=10, n_levels=1, seed=0):
    levels, level_weights = List(), List()
    rng = np.random.default_rng(seed)
    levels.append(level_dates)
    level_weights_data = [1]
    n_samples = level_dates.shape[0]
    for level in range(n_levels):
        values = rng.integers(n_categories, size=n_samples)
        level = pd.get_dummies(values).to_numpy().astype(bool)
        levels.append(level)
        level_weights_data += [1]
    
    level_weights = List(level_weights_data)

    return levels, level_weights


# RMSE per level to evaluate final performance per level
@njit(fastmath = True, parallel = True)
def rmse_levels(yhat, y, levels):
    error = (yhat - y)
    dates = levels[0].T
    n_levels = len(levels) + 1
    n_dates = dates.shape[0]
    n_categories = np.zeros((n_dates, len(levels) - 1))
    loss = np.zeros((n_dates, n_levels))
    for i in prange(n_dates):
        date = dates[i]
        errord = error[date] * 1.
        # Level loss per day
        for j, level in enumerate(levels[1:]):
            leveld = level[date] * 1.
            # Not every category is represented on each day, so we keep track of actual no. categories per level per day
            n_categories[i, j] = np.count_nonzero(np.count_nonzero(leveld, axis=0))
            loss[i, j + 1] = np.sum((errord @ leveld)**2)
        # Daily loss
        loss[i, 0] = (1 / n_dates) * np.sum(errord)**2

    n_categories = n_categories.sum(axis=0)
    loss = np.sum(loss, axis=0)
    loss[1:-1] /= n_categories
    # The last loss is the most granular per-product loss
    loss[-1] = np.mean(error**2)
        
    return np.sqrt(loss)
#%% Lightgbm - training parameters
params = {'min_split_gain':0,
          'min_data_in_leaf':20,
          'max_depth':-1,
          'max_bin':255,
          'max_leaves':31,
          'learning_rate':0.1,
          'n_estimators':1000,
          'verbose':1,
          'feature_fraction':1.0,
          'bagging_fraction':1.0,
          'bagging_freq':1,
          'seed':0,
          'lambda':0,
          'objective': None,
          'metric': 'hierarchical_se',
          'device':'cpu'}

path = 'src/exp1/'
algorithm = 'lightgbm_randomlevels'
dataset = 'm5'
loss_function = params['metric']
experiment_name = f'{algorithm}_{dataset}_{loss_function}'
level_weights = List([1, 1])
drop_cols = ['date']
cat_cols = ['item_id_enc', 'dept_id_enc', 'cat_id_enc']
level_cols = ['date']
#%% Validation loop
forecast_day = 0
metrics = []
X, y = create_forecastset(subset, cols_unknown, cols_known, forecast_day)
train_last_date = '2016-03-27'
val_first_date = '2016-03-28'
val_last_date = '2016-04-24'
# Create training and validation set
X_train, y_train = X[X.date <= train_last_date], y[y.date <= train_last_date]
X_val, y_val = X[(X.date >= val_first_date) & (X.date <= val_last_date)], y[(y.date >= val_first_date) & (y.date <= val_last_date)]
# Create levels for weighted squared error
levels_train = create_levels(X_train[level_cols])
levels_valid = create_levels(X_val[level_cols])
# Create X, y tuples
X_train, y_train = X_train.drop(columns=drop_cols), y_train.drop(columns=drop_cols)
X_val, y_val = X_val.drop(columns=drop_cols), y_val.drop(columns=drop_cols)
# Construct LGB dataset
params['bin_construct_sample_cnt'] = len(X_train)
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
valid_set = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)
train_set.levels = levels_train
train_set.level_weights = level_weights
valid_set.levels = levels_valid
valid_set.level_weights = level_weights    
# Train model
model = lgb.train(params,
                    train_set = train_set,
                    valid_sets = valid_set,
                    early_stopping_rounds=20,
                    fobj = hierarchical_obj_se,
                    feval = hierarchical_eval_se)
params['n_estimators'] = model.best_iteration + 1
#%% Test
forecast_day = 0
X, y = create_forecastset(subset, cols_unknown, cols_known, forecast_day)
train_last_date = '2016-04-24'
test_first_date = '2016-04-25'
# Create training and validation set
X_train, y_train = X[X.date <= train_last_date], y[y.date <= train_last_date]
X_test, y_test = X[(X.date >= test_first_date)], y[(y.date >= test_first_date)]
# Create levels for weighted squared error
levels_train = create_levels(X_train[level_cols])
test_level_cols = ['date', 'cat_id_enc', 'dept_id_enc']
levels_test = create_levels(X_test[test_level_cols])
# Create X, y tuples
X_train, y_train = X_train.drop(columns=drop_cols), y_train.drop(columns=drop_cols)
X_test, y_test = X_test.drop(columns=drop_cols), y_test.drop(columns=drop_cols)
# Construct LGB dataset
params['bin_construct_sample_cnt'] = len(X_train)
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
train_set.levels = levels_train
train_set.level_weights = level_weights
# Train model
model = lgb.train(params, train_set = train_set, fobj = hierarchical_obj_se)
# Predict
yhat = np.clip(model.predict(X_test), 0,  1e9)
rmse_level = rmse_levels(yhat, y_test.values.astype(np.float64).squeeze(), levels_test)
#%% Logging
metrics = []
metrics.append((train_last_date, test_first_date, params['n_estimators']) + tuple(rmse_level))
# Create dataframe and save
df_test_metrics = pd.DataFrame(metrics, columns=['train_last_date', 'test_first_date', 'best_estimators']\
                                                  + [f'rmse_{col}' for col in test_level_cols] + ['rmse_product'])
df_test_metrics.to_csv(path + experiment_name + '.csv')
shutil.copy('src/model_lightgbm_random.py', path + experiment_name + '.py')
model.save_model(path + experiment_name + '.model')