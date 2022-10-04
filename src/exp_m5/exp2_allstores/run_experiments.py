#%% Read packages
import pandas as pd
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods
from src.exp_m5 import read_m5, create_forecast_set, exp_m5_globalall, exp_m5_sepagg, exp_m5_globalbottomup
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
# Other experiment settings
folder = 'exp2_allstores'
n_seeds = 10
default_params = {'seed': 0,
                  'n_estimators': 2000,
                  'n_trials': 100,
                  'learning_rate': 0.1,
                  'verbosity': -1,
                  'tuning': True,
                  'n_validation_sets': 3,
                  'max_levels_random': 2,
                  'max_categories_per_random_level': 1000,
                  'n_days_test': 28}
#%% Read data
df = read_m5(store_level=False)
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, 
                            aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Create forecast set
X, Xind, targets = create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day=0)
#%% Setting 1: global models for all time series
experiments_global = [{'exp_name':'globalall_objse_evalmse'}]
for experiment in experiments_global:
    df_result = pd.DataFrame()
    for seed in range(n_seeds):
        exp_name = experiment['exp_name']
        params = default_params.copy()
        forecast_seed =  exp_m5_globalall(X, Xind, targets, target, time_index, end_train, 
                                         exp_name=exp_name, params=params, exp_folder=folder,
                                         seed=seed)
        # Apply reconciliation methods
        forecasts_test = forecast_seed.loc[:, start_test:]
        forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, targets.loc[:, :end_train], forecast_seed.loc[:, :end_train],
                        methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True)
        # Add result to result df
        dfc = pd.concat({f'{seed}': forecasts_methods}, names=['Seed'])
        df_result = pd.concat((df_result, dfc))
    df_result.to_csv(f'./src/exp_m5/{folder}/{exp_name}.csv')
#%% Setting 2: separate model for each aggregation in the hierarchy
# experiments_agg = [{'exp_name':'sepagg_objse_evalmse'}]
# for experiment in experiments_agg:
#     for seed in range(n_seeds):
#         exp_name = experiment['exp_name']
#         params = default_params.copy()
#         forecast_seed =  exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, 
#                                          df_S, exp_name=exp_name, params=params,
#                                          exp_folder=folder, seed=seed)
#         experiment[f'forecast_seed_{seed}'] = forecast_seed
#         # Apply reconciliation methods
#         forecasts_test = forecast_seed.loc[:, start_test:]
#         forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, targets.loc[:, :end_train], forecast_seed.loc[:, :end_train],
#                         methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True)
#         # Store all forecasts
#         experiment[f'forecast_seed_{seed}'] = forecasts_methods
# #%% Setting 3: global models for bottom-up series
# experiments_bu = [
#                     {'exp_name':'bu_objmse_evalmse',
#                     'fobj':'l2',
#                     'feval':'l2'},
#                     {'exp_name':'bu_objmse_evalhmse',
#                     'fobj':'l2',
#                     'feval': 'hierarchical_eval_hmse'},
#                     {'exp_name':'bu_objtweedie_evalmse',
#                     'fobj':'tweedie',
#                     'feval':'l2'},
#                     {'exp_name':'bu_objtweedie_evalhmse',
#                     'fobj':'tweedie',
#                     'feval': 'hierarchical_eval_hmse'},
#                     {'exp_name':'bu_objtweedie_evaltweedie',
#                     'fobj':'tweedie',
#                     'feval': 'tweedie'},
#                     {'exp_name':'bu_objhse_evalhmse',
#                     'fobj': 'hierarchical_obj_se',
#                     'feval': 'hierarchical_eval_hmse'},
#                     {'exp_name':'bu_objhse_evalmse',
#                     'fobj': 'hierarchical_obj_se',
#                     'feval': 'l2'},
#                     {'exp_name':'bu_objrhse_evalhmse',
#                     'fobj': 'hierarchical_obj_se_random',
#                     'feval': 'hierarchical_eval_hmse'},
#                     {'exp_name':'bu_objrhse_evalmse',
#                     'fobj': 'hierarchical_obj_se_random',
#                     'feval': 'l2'},
#                     ]
# # We loop over all the experiments and create forecasts for n_seeds
# for experiment in experiments_bu:
#     df_result = pd.DataFrame()
#     for seed in range(n_seeds):
#         exp_name = experiment['exp_name']
#         fobj = experiment['fobj']
#         feval = experiment['feval']
#         params = default_params.copy()
#         forecast_seed =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, 
#                                                 name_bottom_timeseries, df_S, exp_folder=folder,
#                                                 params=params, exp_name=exp_name,
#                                                 fobj=fobj, feval=feval, seed=seed)
#         forecasts_test = forecast_seed.loc[:, start_test:]
#         forecasts_method = pd.concat({f"{experiment['exp_name']}" : forecasts_test}, names=['Method'])
#         # Add result to result df
#         dfc = pd.concat({f'{seed}': forecasts_method}, names=['Seed'])
#         df_result = pd.concat((df_result, dfc))
#     df_result.to_csv(f'./src/exp_m5/{folder}/{exp_name}.csv')