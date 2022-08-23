#%% Read packages
import pandas as pd
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods, calc_level_method_rmse
from src.exp_m5 import read_m5, create_forecast_set, exp_m5_globalall, exp_m5_globalbottomup
#%% Set aggregations and target
aggregation_cols = ['state_id_enc', 'store_id_enc', 'cat_id_enc', 'dept_id_enc', 'item_id_enc']
aggregations = [['state_id_enc'],
                ['store_id_enc'],
                ['cat_id_enc'],
                ['dept_id_enc'],
                ['state_id_enc', 'cat_id_enc'],
                ['state_id_enc', 'dept_id_enc'],
                ['store_id_enc', 'cat_id_enc'],
                ['store_id_enc', 'dept_id_enc'],
                ['item_id_enc'],
                ['item_id_enc', 'state_id_enc']]
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = '2016-04-24'
start_test = '2016-04-25'
#%% Read data
df = read_m5(store_level=False)
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, 
                            aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Create forecast set
X, Xind, targets = create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day=0)
#%% Setting 1: global models for all time series
# 1a: Global model for all time series, Squared Error objective, Mean Squared Error metric 
exp_name = 'exp_m5_globalall_objse'
forecasts = exp_m5_globalall(X, Xind, targets, target, time_index, end_train, exp_name, tuning=True)
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
#%% Apply reconciliation methods
forecasts_test = forecasts.loc[:, start_test:]
forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, targets.loc[:, :end_train], forecasts.loc[:, :end_train],
                    methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True)
#%% Calculate error for all levels and methods
forecasts_bu_objmse_evalmse_test = forecasts_bu_objmse_evalmse.loc[:, start_test:]
forecasts_bu_objmse_evalhse_test = forecasts_bu_objmse_evalhse.loc[:, start_test:]
forecasts_bu_objhse_evalhse_test = forecasts_bu_objhse_evalhse.loc[:, start_test:]
forecasts_bu_objhse_evalmse_test = forecasts_bu_objhse_evalmse.loc[:, start_test:]

df_forecasts_bu_objmse_evalmse = pd.concat({'bu_objmse_evalmse': forecasts_bu_objmse_evalmse_test}, names=['Method'])
df_forecasts_bu_objmse_evalhse = pd.concat({'bu_objmse_evalhse': forecasts_bu_objmse_evalhse_test}, names=['Method'])
df_forecasts_bu_objhse_evalhse = pd.concat({'bu_objhse_evalhse': forecasts_bu_objhse_evalhse_test}, names=['Method'])
df_forecasts_bu_objhse_evalmse = pd.concat({'bu_objhse_evalmse': forecasts_bu_objhse_evalmse_test}, names=['Method'])

forecasts_methods_total = pd.concat((df_forecasts_bu_objmse_evalmse, df_forecasts_bu_objmse_evalhse,
                                      df_forecasts_bu_objhse_evalhse, df_forecasts_bu_objhse_evalmse, 
                                      forecasts_methods), axis=0)

rmse, rel_rmse = calc_level_method_rmse(forecasts_methods_total, targets, base='bu_objmse_evalmse')
#%% Save
rmse.to_csv('rmse_lgbm.csv')
rel_rmse.to_csv('rel_rmse.csv')