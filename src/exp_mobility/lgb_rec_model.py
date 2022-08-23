#%% Read packages
import pandas as pd
import numpy as np
from src.lib import reconcile_forecasts, calc_summing_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple
from hts.utilities.load_data import load_mobility_data
import lightgbm as lgb
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
y_train = X['target'].loc[:'2015-12-31']
X_train = X.drop(columns=['target']).loc[:'2015-12-31']
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
# Make predictions for both train and test set (we need the train residuals for covariance estimation)
yhat = model.predict(X.drop(columns=['target']))
df_yhat = pd.Series(index=Xind, data=yhat)
forecasts = df_yhat.unstack(['date']).loc[actuals.index, X.index.get_level_values('date').unique()]
actuals = actuals.loc[:, X.index.get_level_values('date').unique()]
residuals = (forecasts - actuals)
#%% Bottom-up forecasts
all_aggregations = forecasts.index.get_level_values('Aggregation').unique()
all_aggregations = all_aggregations.drop('None')
forecasts_bu = forecasts.copy()
for agg in all_aggregations:
    forecasts_bu.loc[agg] = (df_S.loc[agg] @ forecasts.loc['None']).values

residuals_bu = (forecasts_bu - actuals)
#%% Reconciliation
def apply_all_reconciliation_methods(forecasts, S, residuals_train):
    
    forecasts_method = pd.concat({'base': forecasts}, names=['Method'])
    cols = forecasts_method.columns
    yhat = forecasts_method.values.astype(np.float64)
    # Apply all reconciliation methods
    methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'mint_lw', 'mint_oas']
    forecasts_methods = []
    forecasts_methods.append(forecasts_method)
    for method in methods:
        ytilde = reconcile_forecasts(yhat, S.values, residuals_train.values, method=method)
        forecasts_method = pd.DataFrame(data=ytilde,
                                        index=forecasts.index, 
                                        columns=cols)
        forecasts_method = pd.concat({f'{method}': forecasts_method}, names=['Method'])
        forecasts_methods.append(forecasts_method)

    forecasts_methods = pd.concat(forecasts_methods)
    return forecasts_methods

# All forecasts
residuals_train = residuals.loc[:, :'2015-12-31']
forecasts_test = forecasts.loc[:, '2016-01-01':]
forecasts_methods = apply_all_reconciliation_methods(forecasts_test, df_S, residuals_train)
# Bottom-up
residuals_bu_train = residuals_bu.loc[:, :'2015-12-31']
forecasts_bu_test = forecasts_bu.loc[:, '2016-01-01':]
forecasts_bu_methods = apply_all_reconciliation_methods(forecasts_bu_test, df_S, residuals_bu_train)
#%% Calculate errors for all methods and levels
def calc_level_method_rmse(forecasts_methods, actuals):
    rmse_index = forecasts_methods.index.droplevel(['Value']).drop_duplicates()
    rmse = pd.DataFrame(index=rmse_index, columns=['RMSE'], dtype=np.float64)
    methods = forecasts_methods.index.get_level_values('Method').unique()
    for method in methods:
        forecasts_method = forecasts_methods.loc[method]
        sq_error = (forecasts_method - actuals.loc[:, forecasts_method.columns])**2
        rmse_current = np.sqrt(sq_error.stack().groupby(['Aggregation']).mean())
        rmse.loc[method, 'RMSE'] = rmse_current.loc[rmse.loc[method, 'RMSE'].index].values

    rmse = rmse.unstack(0)
    rmse.columns = rmse.columns.droplevel(0)
    rmse = rmse[methods].sort_values(by='base', ascending=False)
    rel_rmse = rmse.div(rmse['base'], axis=0)

    return rmse, rel_rmse

rmse, rel_rmse = calc_level_method_rmse(forecasts_methods, actuals)
rmse_bu, rel_rmse_bu = calc_level_method_rmse(forecasts_bu_methods, actuals)