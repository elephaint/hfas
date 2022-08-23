# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:01:40 2022

@author: ospra
"""
#%% Import packages
import numpy as np
import pandas as pd
import time
import shutil
from numba import njit, prange
from numba.typed import List, Dict
from numba.core import types
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from hts.utilities.load_data import load_mobility_data
#%% Load data
hd = load_mobility_data()
data = hd.drop(columns = ['total', 'CH', 'SLU', 'BT', 'OTHER', 'temp', 'precipitation'])
data = data.stack().reset_index()
data.rename(columns = {'level_1':'item', 0:'target'}, inplace=True)
data['group'] = data['item'].str.split('-', 1, expand=True)[0].astype("category")
data['item'] = data['item'].astype("category")
data = data[['starttime', 'group', 'item', 'target']]
data = data.merge(hd.reset_index()[['starttime', 'temp', 'precipitation']], how='left', left_on=['starttime'], right_on=['starttime'])
data.rename(columns = {'starttime':'date'}, inplace=True)
#%% Preprocessing for forecast
def create_forecast_set(data, forecast_day):
    lags = np.arange(1, 8)
    group = data.groupby(['item'])['target']
    for lag in lags:
        data['sales_lag'+str(lag)] = data.groupby(['item'])['target'].shift(lag + forecast_day)
    
    data['weekday'] = data['date'].dt.weekday
    data = data.dropna()

    return data

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

# hierarchical gradient and hessian
@njit(fastmath = True, parallel=True)
def hierarchical_gradhess(yhat, y, levels, level_weights, fobj, params):
    dates = levels[0].T
    n_levels = len(levels) + 1
    n_dates = dates.shape[0]
    level_cnt = np.ones_like(y)
    gradient, hessian = fobj(yhat, y, params, level_cnt)
    gradient, hessian = (level_weights[-1] / n_levels) * gradient, (level_weights[-1] / n_levels) * hessian
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
    # levels = train_data.levels
    level_dates = train_data.levels[0]
    levels, level_weights = create_random_levels(level_dates, seed = train_data.params['seed'])
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
          'learning_rate':0.02,
          'n_estimators':1000,
          'verbose':1,
          'feature_fraction':1.0,
          'bagging_fraction':1.0,
          'bagging_freq':1,
          'seed':0,
          'lambda':0,
        #   'objective': None,
          'objective':'mse',
        #   'metric':'mse',
          'metric': 'hierarchical_se',
          'device':'cpu'}

algorithm = 'lightgbm'
dataset = 'mobility'
experiment = 'mse_hierarchical_eval'
experiment_name = f'{algorithm}_{dataset}_{experiment}'
level_weights = List([1, 1, 1])
drop_cols = ['date', 'target']
cat_cols = ['group', 'item']
level_cols = ['date', 'group']
#%% Validation loop
forecast_day = 0
metrics = []
df = create_forecast_set(data, 0)
n_training_days = 3 * 366 
n_validation_sets = 6
n_days_per_validation_set = 31
n_validation_days = n_validation_sets * n_days_per_validation_set 
n_test_days = 90
best_estimators = np.zeros(n_validation_sets)
max_date = df['date'].max()
test_date = max_date - pd.Timedelta(n_test_days - 1, 'd')
for i, j in zip(range(n_validation_sets, 0, -1), range(n_validation_sets)):
    train_date = max_date - pd.Timedelta(n_training_days + i * n_days_per_validation_set + n_test_days - 1, 'd')
    validation_date = max_date - pd.Timedelta(i * n_days_per_validation_set + n_test_days - 1, 'd')
    df_train = df[(df['date'] >= train_date) & (df['date'] < validation_date)]
    df_validate = df[(df['date'] >= validation_date) & (df['date'] < (validation_date + pd.Timedelta(n_days_per_validation_set, 'd')))]
    # Some assertions to check that we create correct non-overlapping sets
    # assert df_train['date'].nunique() == n_training_days
    assert df_validate['date'].nunique() == n_days_per_validation_set
    assert df_validate['date'].min() > df_train['date'].max()
    assert df_validate['date'].min() - df_train['date'].max() == pd.Timedelta(1, 'd')
    assert df_validate['date'].max() + pd.Timedelta((i - 1) * n_days_per_validation_set + 1, 'd') == test_date
    # Create levels for weighted squared error
    levels_train = create_levels(df_train[level_cols])
    levels_valid = create_levels(df_validate[level_cols])
    # Create X, y
    X_train, X_validate = df_train.drop(columns=drop_cols), df_validate.drop(columns=drop_cols)
    y_train, y_validate = df_train['target'], df_validate['target']
    # Construct LGB dataset
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_validate, y_validate)
    train_set.levels = levels_train
    train_set.level_weights = level_weights
    valid_set.levels = levels_valid
    valid_set.level_weights = level_weights    
    # Train model
    model = lgb.train(params,
                        train_set = train_set,
                        valid_sets = valid_set,
                        early_stopping_rounds=20,
                        # fobj = hierarchical_obj_se,
                        feval = hierarchical_eval_se)
    # model = lgb.train(params,
    #                     train_set = train_set,
    #                     valid_sets = valid_set,
    #                     early_stopping_rounds=20)                        
    best_estimators[j] = model.best_iteration + 1

#%% Test set
params['n_estimators'] = int(np.mean(best_estimators))
df_test = df[df['date'] >= test_date].copy()
assert df_test['date'].nunique() == n_test_days
train_date = max_date - pd.Timedelta(n_training_days + n_test_days - 1, 'd')
df_train = df[(df['date'] >= train_date) & (df['date'] < test_date)]
assert df_test['date'].min() > df_train['date'].max()
assert df_test['date'].min() - df_train['date'].max() == pd.Timedelta(1, 'd')
# Create levels for weighted squared error
levels_train = create_levels(df_train[level_cols])
levels_test = create_levels(df_test[level_cols])
test_level_cols = ['date', 'group']
levels_test = create_levels(df_test[test_level_cols])
# Create X, y
X_train, X_test = df_train.drop(columns=drop_cols), df_test.drop(columns=drop_cols)
y_train, y_test = df_train['target'], df_test['target']
# Construct LGB dataset
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
train_set.levels = levels_train
train_set.level_weights = level_weights
# Train model
# model = lgb.train(params, train_set = train_set, fobj = hierarchical_obj_se)
model = lgb.train(params, train_set = train_set)
yhat_test = model.predict(X_test)
df_test['target_prediction'] = yhat_test
rmse_level = rmse_levels(yhat_test, y_test.values.astype(np.float64).squeeze(), levels_test)
#%% Plot predictions
rng = np.random.default_rng()
series = rng.choice(df_test['item'].unique())
df_train_plot = df_train.set_index(['item', 'date'])
df_test_plot = df_test.set_index(['item', 'date'])
# plt.plot(df_train_plot.loc[series, 'sales'], label='Train set')
plt.plot(df_test_plot.loc[series, 'target'], label='Test set')
plt.plot(df_test_plot.loc[series, 'target_prediction'], label='Test predictions')
#%% Logging
metrics = []
metrics.append((experiment_name, params['n_estimators']) + tuple(rmse_level))
df_cmetrics = pd.DataFrame(metrics, columns=['experiment', 'n_estimators']\
                                                + [f'rmse_{col}' for col in test_level_cols] + ['rmse_item'])
# Create dataframe and save
metrics_path = 'experiment_results.csv'
metrics = Path(metrics_path)
if metrics.is_file():
    df_metrics = pd.read_csv(metrics_path, index_col=0)
    df_metrics = pd.concat((df_metrics, df_cmetrics), axis=0, ignore_index=True)
else:
    df_metrics = df_cmetrics
df_metrics.to_csv(metrics_path)
shutil.copy(Path(__file__).name, 'models/' + experiment_name + '.py')
model.save_model('models/' + experiment_name + '.model')