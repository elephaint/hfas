#%% Read packages
import pandas as pd
from hierts.reconciliation import apply_reconciliation_methods, hierarchy_temporal, hierarchy_cross_sectional
from src.exp_m5 import read_m5, create_forecast_set, exp_m5_globalall, exp_m5_sepagg, exp_m5_globalbottomup
#%% Set aggregations and target
cross_sectional_aggregations = [['cat_id_enc'],
                                ['dept_id_enc'],
                                ['item_id_enc']]
temporal_aggregations = [['year'],
                         ['year', 'month'],
                         ['year', 'week']]
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = '2016-04-24'
start_test = '2016-04-25'
# Other experiment settings
exp_folder = 'exp1_storeid=0_temporal'
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
                  'n_days_test': 28,
                  'n_years_train': 4}
#%% Read data
df = read_m5(store_level=True, store_id=0)
# Add columns for temporal hierarchies
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Remove the double in the item_id_enc shape
df_Sc = df_Sc.drop('item_id_enc') 
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
# Create forecast set
aggregation_cols = [col for cols in cross_sectional_aggregations for col in cols]
df = df.drop(columns = ['week', 'year', 'month', 'day'])
X, Xind, targets = create_forecast_set(df, df_Sc, aggregation_cols, time_index, target, forecast_day=0)
#%% Setting 1: global models for all time series
experiments_global = [{'exp_name':'globalall_objse_evalmse'}]
for experiment in experiments_global:
    df_result = pd.DataFrame()
    df_result_timings = pd.DataFrame()
    for seed in range(n_seeds):
        exp_name = experiment['exp_name']
        params = default_params.copy()
        forecast_seed, t_train_seed, t_predict_seed =  exp_m5_globalall(X, Xind, targets, target, time_index, end_train, df_Sc, df_St, 
                                         exp_name=exp_name, params=params, exp_folder=exp_folder,
                                         seed=seed)
        # Apply reconciliation methods
        forecasts_test = forecast_seed.loc[:, start_test:]
        forecasts_methods, t_reconciliation_seed = apply_reconciliation_methods(forecasts_test, df_Sc, targets.loc[:, :end_train], forecast_seed.loc[:, :end_train],
                        methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True, return_timing=True)
        # Add result to result df
        dfc = pd.concat({f'{seed}': forecasts_methods}, names=['Seed'])
        df_result = pd.concat((df_result, dfc))
        # Add timings to timings df
        df_seed = pd.DataFrame({'t_train':t_train_seed, 't_predict':t_predict_seed}, index=[seed])
        df_reconciliation = pd.DataFrame(t_reconciliation_seed, index=[seed])
        df_result_timings = pd.concat((df_result_timings, pd.concat((df_seed, df_reconciliation), axis=1)))
    # Save
    df_result.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}.csv')
    df_result_timings.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}_timings.csv')
#%% Setting 2: separate model for each aggregation in the hierarchy
experiments_agg = [{'exp_name':'sepagg_objmse_evalmse'}]
for experiment in experiments_agg:
    df_result = pd.DataFrame()
    df_result_timings = pd.DataFrame()
    for seed in range(n_seeds):
        exp_name = experiment['exp_name']
        params = default_params.copy()
        forecast_seed, t_train_seed, t_predict_seed =  exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, 
                                         df_Sc, df_St, exp_name=exp_name, params=params,
                                         exp_folder=exp_folder, seed=seed)
        experiment[f'forecast_seed_{seed}'] = forecast_seed
        # Apply reconciliation methods
        forecasts_test = forecast_seed.loc[:, start_test:]
        forecasts_methods, t_reconciliation_seed = apply_reconciliation_methods(forecasts_test, df_Sc, targets.loc[:, :end_train], forecast_seed.loc[:, :end_train],
                        methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'erm'], positive=True, return_timing=True)
        # Add result to result df
        dfc = pd.concat({f'{seed}': forecasts_methods}, names=['Seed'])
        df_result = pd.concat((df_result, dfc))
        # Add timings to timings df
        df_seed = pd.DataFrame({'t_train':t_train_seed, 't_predict':t_predict_seed}, index=[seed])
        df_reconciliation = pd.DataFrame(t_reconciliation_seed, index=[seed])
        df_result_timings = pd.concat((df_result_timings, pd.concat((df_seed, df_reconciliation), axis=1)))
    # Save
    df_result.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}.csv')
    df_result_timings.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}_timings.csv')

#%% Setting 3: global models for bottom-up series
experiments_bu = [
                    {'exp_name':'bu_objmse_evalmse',
                    'sobj':'l2',
                    'seval':'l2'},
                    # {'exp_name':'bu_objmse_evalhmse',
                    # 'sobj':'l2',
                    # 'seval': 'hierarchical_eval_hmse'},
                    # {'exp_name':'bu_objtweedie_evalmse',
                    # 'sobj':'tweedie',
                    # 'seval':'l2'},
                    # {'exp_name':'bu_objtweedie_evalhmse',
                    # 'sobj':'tweedie',
                    # 'seval': 'hierarchical_eval_hmse'},
                    # {'exp_name':'bu_objtweedie_evaltweedie',
                    # 'sobj':'tweedie',
                    # 'seval': 'tweedie'},
                    {'exp_name':'bu_objhse_evalhmse',
                    'sobj': 'hierarchical_obj_se',
                    'seval': 'hierarchical_eval_hmse'},
                    # {'exp_name':'bu_objhse_evalhmse_withtemp',
                    # 'sobj': 'hierarchical_obj_se_withtemp',
                    # 'seval': 'hierarchical_eval_hmse_withtemp'},
                    # {'exp_name':'bu_objhse_evalmse',
                    # 'sobj': 'hierarchical_obj_se',
                    # 'seval': 'l2'},
                    # {'exp_name':'bu_objrhse_evalhmse',
                    # 'sobj': 'hierarchical_obj_se_random',
                    # 'seval': 'hierarchical_eval_hmse'},
                    # {'exp_name':'bu_objrhse_evalmse',
                    # 'sobj': 'hierarchical_obj_se_random',
                    # 'seval': 'l2'},
                    ]
# We loop over all the experiments and create forecasts for n_seeds
for experiment in experiments_bu:
    df_result = pd.DataFrame()
    df_result_timings = pd.DataFrame()
    for seed in range(n_seeds):
        exp_name = experiment['exp_name']
        sobj = experiment['sobj']
        seval = experiment['seval']
        params = default_params.copy()
        forecasts_test, t_train_seed, t_predict_seed =  exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, start_test, 
                                                name_bottom_timeseries, df_Sc, df_St, exp_folder=exp_folder,
                                                params=params, exp_name=exp_name,
                                                sobj=sobj, seval=seval, seed=seed)
        forecasts_method = pd.concat({f"{experiment['exp_name']}" : forecasts_test}, names=['Method'])
        # Add result to result df
        dfc = pd.concat({f'{seed}': forecasts_method}, names=['Seed'])
        df_result = pd.concat((df_result, dfc))
        # Add timings to timing df
        df_seed = pd.DataFrame({'t_train':t_train_seed, 't_predict':t_predict_seed}, index=[seed])
        df_result_timings = pd.concat((df_result_timings, df_seed))
    # Save
    df_result.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}.csv')
    df_result_timings.to_csv(f'./src/exp_m5/{exp_folder}/{exp_name}_timings.csv')