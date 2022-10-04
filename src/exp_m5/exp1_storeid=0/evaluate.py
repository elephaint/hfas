#%% Import packages
import pandas as pd
from hierts.reconciliation import calc_summing_matrix, calc_level_method_rmse
from src.exp_m5 import read_m5, create_forecast_set
#%% Load data
aggregation_cols = ['cat_id_enc', 'dept_id_enc', 'item_id_enc']
aggregations = [['cat_id_enc'],
                ['dept_id_enc']]
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = '2016-04-24'
start_test = '2016-04-25'
n_seeds = 5
df = read_m5(store_level=True, store_id=0)
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, 
                            aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Create forecast set
_, _, targets = create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day=0)
#%% Load experimental results
folder = './src/exp_m5/exp1_storeid=0'
experiments = ['globalall_objse_evalmse', 
                'bu_objmse_evalmse', 'bu_objmse_evalhmse', 
                'bu_objhse_evalhmse', 'bu_objhse_evalmse',
                'bu_objrhse_evalhmse', 'bu_objrhse_evalmse',
                'bu_objtweedie_evalmse', 'bu_objtweedie_evalhmse', 'bu_objtweedie_evaltweedie',
                ]

base='bu_objmse_evalmse'
rmse = pd.DataFrame()
for seed in range(n_seeds):
    df_result = pd.DataFrame()
    for experiment in experiments:
        df = pd.read_csv(f'{folder}/{experiment}.csv', index_col=[0, 1, 2, 3])
        df_result = pd.concat((df_result, df.loc[seed]))

    df_result.columns = df_result.columns.map(pd.to_datetime)
    rmse_seed, _ = calc_level_method_rmse(df_result, targets, base=base)
    rmse_seed = pd.concat({f'{seed}': rmse_seed}, names=['Seed'])
    rmse = pd.concat((rmse, rmse_seed))

rmse_mean = rmse.groupby(['Aggregation']).mean()
index_cols = list(rmse_mean.index.drop('All series')) + ['All series']
rmse_mean = rmse_mean.reindex(index = index_cols)
rel_rmse = rmse_mean.div(rmse_mean[base], axis=0)

#%% Compute scores
rmse, rel_rmse = calc_level_method_rmse(df_result, targets, base='bu_objmse_evalmse')
#%% Save
rmse.to_csv(f'{folder}/rmse_lgbm_hier.csv')
rel_rmse.to_csv(f'{folder}/rel_rmse_lgbm_hier.csv')