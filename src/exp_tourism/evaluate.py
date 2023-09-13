#%% Import packages
import pandas as pd
from hierts.reconciliation import hierarchy_cross_sectional, hierarchy_temporal, calc_level_method_error
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
from helper_functions import read_tourism, get_aggregations, create_forecast_set
#%% Load data
learning_rate = 0.05
exp_folder = f"exp1_lr{learning_rate}"
assert CURRENT_PATH.joinpath(exp_folder).is_dir()
cross_sectional_aggregations, temporal_aggregations = get_aggregations()
time_index = 'Quarter'
target = 'Trips'
name_bottom_timeseries = 'bottom_timeseries'
end_train = pd.to_datetime("2015Q4")
start_test = pd.to_datetime("2016Q1")
# Other experiment settings
n_seeds = 10
#%% Read data
df = read_tourism()
# Add columns for temporal hierarchies
df["year"] = df["Quarter"].dt.year
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
# Create forecast set
aggregation_cols = list(dict.fromkeys([col for cols in cross_sectional_aggregations for col in cols]))
df = df.drop(columns = ['year'])
_, _, targets = create_forecast_set(df, df_Sc, aggregation_cols, time_index, target)
#%% Load experimental results
experiments = [ 
                'globalall_objse_evalmse', 
                'bu_objmse_evalmse',
                'bu_objmse_evalhmse', 
                'bu_objhse_evalhmse', 
                'bu_objhse_evalmse',
                'bu_objrhse_evalhmse',
                'bu_objhse_evalhmse_withtemp',
                'bu_objhse_evalhmse_withtemponly',
                'sepagg_objse_evalmse',
                'Naive_',
                'SeasonalNaive_',
                'AutoETS_',
                'AutoARIMA_',
                'AutoTheta_',
                ]
# Load results
df_result = pd.DataFrame()
for experiment in experiments:
    df = pd.read_parquet(str(CURRENT_PATH.joinpath(f"{exp_folder}/{experiment}.parquet")))
    scenario = experiment[:experiment.find('_')]
    if scenario == 'bu':
        df = df.rename(index = {df.index.get_level_values(1).unique()[0]:experiment}, level=1)
    df.columns = pd.DatetimeIndex(df.columns)
    df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
    df = pd.concat({f"{scenario}": df}, names=['Scenario'])
    df_result = pd.concat((df_result, df))
df_result.columns = df_result.columns.map(pd.to_datetime)
# Calculate error per seed
metric = 'MAE'
error = pd.DataFrame()
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scenarios = ['globalall', 'sepagg', 'SeasonalNaive', 
             'Naive', 'AutoETS', 'AutoTheta', 
             'Naive', 'AutoARIMA', 'bu']

for scenario in scenarios:
    for seed in seeds:
        try:
            df_seed = df_result.loc[(scenario, seed, slice(None), slice(None))]
            error_seed = calc_level_method_error(forecasts_methods=df_seed, actuals=targets, metric=metric)
            error_seed = pd.concat({f'{seed}': error_seed}, names=['Seed'])
            error_seed = pd.concat({f"{scenario}": error_seed}, names=['Scenario'])
            error = pd.concat((error, error_seed))
        except:
            pass
# Aggregate results
error_mean = error.groupby(['Scenario', 'Aggregation']).mean()
error_mean = error_mean.unstack(0).T.swaplevel(0, 1).sort_index(level=0).dropna()
error_std = error.groupby(['Scenario', 'Aggregation']).std()
error_std = error_std.unstack(0).T.swaplevel(0, 1).sort_index(level=0).dropna()
#%% Save
error_mean.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{metric}_mean.csv")))
error_std.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{metric}_std.csv")))
#%% Variance plot
import seaborn as sns
import matplotlib.pyplot as plt

error_allseries = error.loc[(['sepagg', 'bu'], slice(None), 'All series')]
keep_cols = [   'bu_objhse_evalhmse', 
                'bu_objhse_evalhmse_withtemp',
                'bu_objmse_evalmse', 
                'wls_var',
                'mint_shrink'
                ]
renamed_cols = ['Bottom-up: HL/HL', 
                'Bottom-up: HL+/HL+', 
                'Bottom-up: SL/SL', 
                'Sep. agg.: WLS-var', 
                'Sep. agg.: MinT-shrink']
error_allseries = error_allseries[keep_cols]
error_allseries.columns = pd.Index(renamed_cols, name='Method')
error_allseries = error_allseries.stack(0)
error_allseries = pd.DataFrame(error_allseries).rename(columns={0:metric}).sort_index()
error_allseries.reset_index(inplace=True)
error_allseries.drop(columns = 'Scenario', inplace=True)

colors = ['#ff7f0e','#1f77b4','#2ca02c']
sns.set_palette(sns.color_palette(colors))
fig, axes = plt.subplots(1, 1)
for i, ax in enumerate(fig.axes):
    sns.boxplot(ax = ax, y=metric, x='Method', data=error_allseries, width=1, showfliers=False)
    ax.set_title(label=f'{metric} - All series', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.set_ylabel(ylabel = f'{metric}', fontsize=12)
    # ax.legend_.remove()
handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc = 'lower center', ncol=3)
leg.get_frame().set_linewidth(0.0)
plt.xticks(rotation=45)
fig.tight_layout()

import numpy as np
np.count_nonzero(df_seed)