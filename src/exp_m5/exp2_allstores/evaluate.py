#%% Import packages
import pandas as pd
from hierts.reconciliation import calc_summing_matrix, calc_level_method_rmse
from src.exp_m5 import read_m5, create_forecast_set
#%% Load data
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
n_seeds = 10
df = read_m5(store_level=False)
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, 
                            aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Create forecast set
X, Xind, targets = create_forecast_set(df, df_S, aggregation_cols, time_index, target, forecast_day=0)
#%% Load experimental results
folder = './src/exp_m5/exp2_allstores'
experiments = [ 
                'globalall_objse_evalmse', 
                'bu_objmse_evalmse',
                # 'bu_objmse_evalhmse', 
                # 'bu_objtweedie_evalmse', 
                # 'bu_objtweedie_evalhmse',
                # 'bu_objtweedie_evaltweedie',
                'bu_objhse_evalhmse', 
                # 'bu_objhse_evalhmse_moreiters', 
                # 'bu_objhse_evalhmse_nofeaturefraction', 
                # 'bu_objhse_evalhmse_mildfeaturefraction', 
                # 'bu_objhse_evalhmse_nobagging',
                'bu_objhse_evalmse',
                'bu_objrhse_evalhmse',
                # 'bu_objrhse_evalmse',
                # 'sepagg_objse_evalmse'
                ]

base='bu_objmse_evalmse'
# Load results
df_result = pd.DataFrame()
for experiment in experiments:
    df = pd.read_parquet(f'{folder}/{experiment}.parquet')
    df = df.rename(index = {df.index.get_level_values(1).unique()[0]:experiment}, level=1)
    df.columns = pd.DatetimeIndex(df.columns)
    df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
    scenario = experiment[:experiment.find('_')]
    df = pd.concat({f"{scenario}": df}, names=['Scenario'])
    df_result = pd.concat((df_result, df))
df_result.columns = df_result.columns.map(pd.to_datetime)
# Calculate rmse per seed
rmse = pd.DataFrame()
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scenarios = ['globalall', 'bu']
for scenario in scenarios:
    for seed in seeds:
        df_seed = df_result.loc[(scenario, seed, slice(None), slice(None))]
        base = df_seed.index.get_level_values(0).unique()[0]
        rmse_seed, _ = calc_level_method_rmse(df_result.loc[(scenario, seed, slice(None), slice(None))], targets, base=base)
        rmse_seed = pd.concat({f'{seed}': rmse_seed}, names=['Seed'])
        rmse_seed = pd.concat({f"{scenario}": rmse_seed}, names=['Scenario'])
        rmse = pd.concat((rmse, rmse_seed))
# Aggregate results
rmse_mean = rmse.groupby(['Scenario', 'Aggregation']).mean()
rmse_mean = rmse_mean.unstack(0).T.swaplevel(0, 1).sort_index(level=0).dropna()
rmse_std = rmse.groupby(['Scenario', 'Aggregation']).std()
rmse_std = rmse_std.unstack(0).T.swaplevel(0, 1).sort_index(level=0).dropna()
#%% Save
rmse_mean.to_csv(f'{folder}/rmse_lgbm_hier.csv')
rmse_std.to_csv(f'{folder}/rmse_std_lgbm_hier.csv')
# rel_rmse.to_csv(f'{folder}/rel_rmse_lgbm_hier.csv')
#%% Variance plot
import seaborn as sns
import matplotlib.pyplot as plt

rmse_allseries = rmse.loc[(slice(None), slice(None), 'All series')]
keep_cols = [   'bu_objhse_evalhmse', 
                'bu_objhse_evalmse',
                'bu_objrhse_evalhmse',
                'bu_objmse_evalhmse', 
                'bu_objmse_evalmse', 
                'bu_objtweedie_evalmse', 
                'wls_var',
                'mint_shrink'
                ]
renamed_cols = ['Bottom-up: HL/HL', 'Bottom-up: HL/SL', 'Bottom-up: RHL/HL', 'Bottom-up: SL/HL', 'Bottom-up: SL/SL', 'Bottom-up: TL/SL', 'Global: WLS-var', 'Global: MinT-shrink']
rmse_allseries = rmse_allseries[keep_cols]
rmse_allseries.columns = pd.Index(renamed_cols, name='Method')
rmse_allseries = rmse_allseries.stack(0)
rmse_allseries = pd.DataFrame(rmse_allseries).rename(columns={0:'RMSE'}).sort_index()
rmse_allseries.reset_index(inplace=True)
rmse_allseries.drop(columns = 'Scenario', inplace=True)

colors = ['#ff7f0e','#1f77b4','#2ca02c']
sns.set_palette(sns.color_palette(colors))
fig, axes = plt.subplots(1, 1)
for i, ax in enumerate(fig.axes):
    sns.boxplot(ax = ax, y="RMSE", x='Method', data=rmse_allseries, width=1, showfliers=False)
    ax.set_title(label='RMSE - All series', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.set_ylabel(ylabel = 'RMSE', fontsize=12)
    # ax.legend_.remove()
handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc = 'lower center', ncol=3)
leg.get_frame().set_linewidth(0.0)
plt.xticks(rotation=45)
fig.tight_layout()