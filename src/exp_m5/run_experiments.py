#%% Read packages
import pandas as pd
import numpy as np
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods, calc_level_method_rmse
from src.exp_m5 import read_m5, create_forecast_set, exp_m5_globalall, exp_m5_sepagg, exp_m5_globalbottomup
#%% Set aggregations and target
aggregation_cols = ['cat_id_enc', 'dept_id_enc', 'item_id_enc']
aggregations = [['cat_id_enc'],
                ['dept_id_enc']]
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = '2016-04-24'
start_test = '2016-04-25'
#%% Read data
df = read_m5(store_level=True, store_id=0)
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, 
                            aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Create forecast set
X, Xind, targets = create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day=0)
#%% Setting 1: global models for all time series
# 1a: Global model for all time series, Squared Error objective, Mean Squared Error metric 
exp_name = 'exp_m5_globalall_objse'
forecasts = exp_m5_globalall(X, Xind, targets, target, time_index, end_train, exp_name, tuning=True)
# 1b: Global model for all time series, Tweedie objective, Mean Squared Error metric
exp_name = 'exp_m5_globalall_objtweedie'
forecasts = exp_m5_globalall(X, Xind, targets, target, time_index, end_train, exp_name, tuning=True)
#%% Setting 2: separate model for each aggregation in the hierarchy
# 2a: Separate model per level, Squared Error objective, Mean Squared Error metric
exp_name = 'exp_m5_sepagg_objse'
forecasts = exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, df_S, exp_name, tuning=True)
# 2b: Separate model per level, Tweedie objective, Mean Squared Error metric
exp_name = 'exp_m5_sepagg_objtweedie'
forecasts = exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, df_S, exp_name, tuning=True)
#%% Setting 3: global models for bottom-up series
# 3a: Global model for bottom-level time series, Squared Error objective, Mean Squared Error metric 
exp_name = 'exp_m5_globalbottomup_objmse_evalmse'
forecasts_bu_objmse_evalmse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj=None, feval=None)
# 3b: Global model for bottom-level time series, Squared Error objective, Hierarchical Squared Error metric
exp_name = 'exp_m5_globalbottomup_objmse_evalhse'
forecasts_bu_objmse_evalhse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj=None, feval='hierarchical_eval_mse')
# 3c: Global model for bottom-level time series, Hierarchical Squared Error objective, Hierarchical Squared Error metric
exp_name = 'exp_m5_globalbottomup_objhse_evalhse'
forecasts_bu_objhse_evalhse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj='hierarchical_obj_se', feval='hierarchical_eval_mse')
# 3d: Global model for bottom-level time series, Hierarchical Squared Error objective, Mean Squared Error metric
exp_name = 'exp_m5_globalbottomup_objhse_evalmse'
forecasts_bu_objhse_evalmse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj='hierarchical_obj_se', feval=None)
# 3e: Global model for bottom-level time series, Random Hierarchical Squared Error objective, Hierarchical Squared Error metric
exp_name = 'exp_m5_globalbottomup_objrhse_evalhse'
forecasts_bu_objrhse_evalhse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj='hierarchical_obj_se_random', feval='hierarchical_eval_mse')                                        
# 3f: Global model for bottom-level time series, Random Hierarchical Squared Error objective, Hierarchical Squared Error metric
exp_name = 'exp_m5_globalbottomup_objrhse_evalmse'
forecasts_bu_objrhse_evalmse =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
                                        name_bottom_timeseries, df_S, exp_name, tuning=True, fobj='hierarchical_obj_se_random', feval=None) 
#%% Apply reconciliation methods
forecasts_test = forecasts.loc[:, start_test:]
forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, targets.loc[:, :end_train], forecasts.loc[:, :end_train],
                    methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True)
#%% Calculate error for all levels and methods
forecasts_bu_objmse_evalmse_test = forecasts_bu_objmse_evalmse.loc[:, start_test:]
forecasts_bu_objmse_evalhse_test = forecasts_bu_objmse_evalhse.loc[:, start_test:]
forecasts_bu_objhse_evalhse_test = forecasts_bu_objhse_evalhse.loc[:, start_test:]
forecasts_bu_objhse_evalmse_test = forecasts_bu_objhse_evalmse.loc[:, start_test:]
forecasts_bu_objrhse_evalhse_test = forecasts_bu_objrhse_evalhse.loc[:, start_test:]
forecasts_bu_objrhse_evalmse_test = forecasts_bu_objrhse_evalmse.loc[:, start_test:]

df_forecasts_bu_objmse_evalmse = pd.concat({'bu_objmse_evalmse': forecasts_bu_objmse_evalmse_test}, names=['Method'])
df_forecasts_bu_objmse_evalhse = pd.concat({'bu_objmse_evalhse': forecasts_bu_objmse_evalhse_test}, names=['Method'])
df_forecasts_bu_objhse_evalhse = pd.concat({'bu_objhse_evalhse': forecasts_bu_objhse_evalhse_test}, names=['Method'])
df_forecasts_bu_objhse_evalmse = pd.concat({'bu_objhse_evalmse': forecasts_bu_objhse_evalmse_test}, names=['Method'])
df_forecasts_bu_objrhse_evalhse = pd.concat({'bu_objrhse_evalhse': forecasts_bu_objrhse_evalhse_test}, names=['Method'])
df_forecasts_bu_objrhse_evalmse = pd.concat({'bu_objrhse_evalmse': forecasts_bu_objrhse_evalmse_test}, names=['Method'])

forecasts_methods_total = pd.concat((df_forecasts_bu_objmse_evalmse, df_forecasts_bu_objmse_evalhse,
                                      df_forecasts_bu_objhse_evalhse, df_forecasts_bu_objhse_evalmse, 
                                      df_forecasts_bu_objrhse_evalhse, df_forecasts_bu_objrhse_evalmse, forecasts_methods), axis=0)

rmse, rel_rmse = calc_level_method_rmse(forecasts_methods_total, targets, base='bu_objmse_evalmse')
#%% Save
rmse.to_csv('src/exp_m5/rmse_lgbm.csv')
rel_rmse.to_csv('src/exp_m5/rel_rmse.csv')