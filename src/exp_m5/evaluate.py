#%% Import packages
import pandas as pd
from hierts.reconciliation import hierarchy_cross_sectional, hierarchy_temporal, calc_level_method_error
from pathlib import Path
import numpy as np
CURRENT_PATH = Path(__file__).parent
from helper_functions import read_m5, get_aggregations, create_forecast_set
#%% Load data
store_id = 0
learning_rate = 0.05
# store_level = True
# exp_folder = f"exp1_storeid={store_id}/lr{learning_rate}"
store_level = False
exp_folder = f"exp2_allstores/lr{learning_rate}"
assert CURRENT_PATH.joinpath(exp_folder).is_dir()
cross_sectional_aggregations, temporal_aggregations = get_aggregations(store_level)
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = pd.to_datetime('2016-04-24')
start_test = pd.to_datetime('2016-04-25')
# Other experiment settings
n_seeds = 10
#%% Read data
df = read_m5(store_level=store_level, store_id=store_id)
# Add columns for temporal hierarchies
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
# Create forecast set
aggregation_cols = list(dict.fromkeys([col for cols in cross_sectional_aggregations for col in cols]))
df = df.drop(columns = ['week', 'year', 'month', 'day'])
_, _, targets = create_forecast_set(df, df_Sc, aggregation_cols, time_index, target, forecast_day=0)
#%% Load experimental results
experiments = [ 
                'globalall_objse_evalmse', 
                'bu_objmse_evalmse',
                'bu_objmse_evalhmse', 
                'bu_objtweedie_evalmse', 
                'bu_objtweedie_evalhmse',
                'bu_objtweedie_evaltweedie',
                'bu_objhse_evalhmse', 
                'bu_objhse_evalmse',
                'bu_objrhse_evalhmse',
                'bu_objhse_evalhmse_withtemp',
                'bu_objhse_evalhmse_withtemponly',
                'bu_objhse_evalhmse_random',
                'sepagg_objse_evalmse',
                'Naive_',
                'SeasonalNaive_',
                'AutoETS_',
                'AutoARIMA_',
                'AutoTheta_',
                'CrostonOptimized_',
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
metrics = ['RMSE', 'MAE']
for metric in metrics:
    error = pd.DataFrame()
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scenarios = ['sepagg', 'SeasonalNaive', 
                'Naive', 'AutoETS', 'AutoTheta', 
                'AutoARIMA', 'CrostonOptimized', 
                'bu', 'globalall']

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
    # Save
    error_mean.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{metric}_mean.csv")))
    error_std.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{metric}_std.csv")))
#%% Variance plot - uncomment to run this
# import seaborn as sns
# import matplotlib.pyplot as plt

# error_allseries = error.loc[(['sepagg', 'bu'], slice(None), 'All series')]
# keep_cols = [   'bu_objhse_evalhmse', 
#                 'bu_objmse_evalmse', 
#                 'bu_objtweedie_evalmse', 
#                 'wls_var',
#                 'mint_shrink'
#                 ]
# renamed_cols = ['Bottom-up: HL/HL', 
#                 'Bottom-up: SL/SL', 
#                 'Bottom-up: TL/SL', 
#                 'Sep. agg.: WLS-var', 
#                 'Sep. agg.: MinT-shrink']
# error_allseries = error_allseries[keep_cols]
# error_allseries.columns = pd.Index(renamed_cols, name='Method')
# error_allseries = error_allseries.stack(0)
# error_allseries = pd.DataFrame(error_allseries).rename(columns={0:metric}).sort_index()
# error_allseries.reset_index(inplace=True)
# error_allseries.drop(columns = 'Scenario', inplace=True)

# colors = ['#ff7f0e','#1f77b4','#2ca02c']
# sns.set_palette(sns.color_palette(colors))
# fig, axes = plt.subplots(1, 1)
# for i, ax in enumerate(fig.axes):
#     sns.boxplot(ax = ax, y=metric, x='Method', data=error_allseries, width=1, showfliers=False)
#     ax.set_title(label=f'{metric} - All series', fontsize=12)
#     ax.tick_params(labelsize=12)
#     ax.spines['top'].set_color('white') 
#     ax.spines['right'].set_color('white')
#     ax.set_ylabel(ylabel = f'{metric}', fontsize=12)
#     # ax.legend_.remove()
# handles, labels = ax.get_legend_handles_labels()
# leg = fig.legend(handles, labels, loc = 'lower center', ncol=3)
# leg.get_frame().set_linewidth(0.0)
# plt.xticks(rotation=45)
# fig.tight_layout()
#%% Forecast plot - uncomment to run this
# import seaborn as sns
# import matplotlib.pyplot as plt

# scenarios = ['sepagg', 'SeasonalNaive', 
#              'Naive', 'AutoETS', 'AutoTheta', 
#              'Naive', 'AutoARIMA', 'CrostonOptimized', 
#              'bu']

# df_forecast = pd.DataFrame()
# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# aggregation = "Total"
# value = "Total"
# for scenario in scenarios:
#     for seed in seeds:
#         try:
#             df_seed = df_result.loc[(scenario, seed, slice(None), aggregation, value)]
#             df_seed = pd.concat({f'{seed}': df_seed}, names=['Seed'])
#             df_seed = pd.concat({f"{scenario}": df_seed}, names=['Scenario'])
#             df_forecast = pd.concat((df_forecast, df_seed))
#         except:
#             pass
# # Aggregate results
# df_forecast_mean = df_forecast.groupby(['Scenario', 'Method']).mean()
# targets_plot = targets.loc[(aggregation, value), df_forecast_mean.columns]
# targets_previous = targets.loc[(aggregation, value), df_forecast_mean.columns[0] - pd.DateOffset(28):df_forecast_mean.columns[0] - pd.DateOffset(1)]

# plot_scenario_methods = [
#                         # ["AutoARIMA", "ols"], 
#                         #  ["AutoETS", "ols"],
#                          ["AutoTheta", "wls_var"],
#                         #  ["CrostonOptimized", "ols"],
#                         #  ["Naive", "ols"],
#                          ["bu", "bu_objhse_evalhmse"],
#                         #  ["sepagg", "mint_shrink"],
#                         #  ["sepagg", "wls_var"],
#                          ]
# # Plot
# fig, axes = plt.subplots(1, 1)
# for i, ax in enumerate(fig.axes):
#     # sns.boxplot(ax = ax, y=metric, x='Method', data=error_allseries, width=1, showfliers=False)
#     sns.lineplot(ax = ax, data=targets_previous)
#     sns.lineplot(ax = ax, data=targets_plot)
#     for (scenario, method) in plot_scenario_methods:
#         sns.lineplot(ax = ax, data=df_forecast_mean.loc[(scenario, method)])
#     ax.set_title(label=f'Forecasts - {aggregation} - {value}', fontsize=12)
#     # ax.tick_params(labelsize=12)
#     # ax.spines['top'].set_color('white') 
#     # ax.spines['right'].set_color('white')
#     ax.set_ylabel(ylabel = 'Forecasts', fontsize=12)
#     # ax.legend_.remove()
# handles, labels = ax.get_legend_handles_labels()
# leg = fig.legend(handles, labels, loc = 'lower center', ncol=3)
# leg.get_frame().set_linewidth(0.0)
# # plt.xticks(rotation=45)
# fig.tight_layout()
#%% Plot RMSE per 7-day time period for selected methods
# Calculate error per seed
error = pd.DataFrame()
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scenarios = ['bu']

experiments = [ 
                'bu_objmse_evalmse',
                'bu_objhse_evalhmse', 
                'bu_objrhse_evalhmse',
                'bu_objhse_evalhmse_withtemp',
                'bu_objhse_evalhmse_withtemponly',
                'bu_objrhse_evalhmse',
                'bu_objhse_evalhmse_random',
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

for scenario in scenarios:
    for seed in seeds:
        try:
            df_seed = df_result.loc[(scenario, seed, slice(None), slice(None))]
            # error_seed = calc_level_method_error(forecasts_methods=df_seed, actuals=targets, metric=metric)
            sq_error = ((df_seed - targets.loc[:, df_seed.columns])**2).stack()
            daily_error = np.sqrt(sq_error.groupby(["Method", time_index]).mean()).unstack(1)
            daily_error.columns = np.arange(len(daily_error.columns)) // 7
            daily_error = daily_error.stack()
            daily_error.index.set_names(["Method", "Days"], inplace=True)
            daily_error = daily_error.groupby(["Method", "Days"]).mean().unstack(1)
            error_seed = pd.concat({f'{seed}': daily_error}, names=['Seed'])
            error_seed = pd.concat({f"{scenario}": error_seed}, names=['Scenario'])
            error = pd.concat((error, error_seed))
        except:
            pass

error = error.groupby(["Method"]).mean()
error /= error.loc['bu_objmse_evalmse']
error.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/bu_error_by_7d_period.csv")))
#%% Plot RMSE per sales bucket - uncomment to create plots
# error = pd.DataFrame()
# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# scenarios = ['bu']

# experiments = [ 
#                 'bu_objmse_evalmse',
#                 # 'bu_objmse_evalhmse', 
#                 'bu_objtweedie_evalmse', 
#                 # 'bu_objtweedie_evalhmse',
#                 # 'bu_objtweedie_evaltweedie',
#                 'bu_objhse_evalhmse', 
#                 # 'bu_objhse_evalhmse_rff',
#                 # 'bu_objhse_evalmse',
#                 # 'bu_objrhse_evalhmse',
#                 # 'bu_objhse_evalhmse_withtemp',
#                 # 'bu_objhse_evalhmse_withtemponly',
#                 # 'sepagg_objse_evalmse',
#                 # 'Naive_',
#                 # 'SeasonalNaive_',
#                 # 'AutoETS_',
#                 # 'AutoARIMA_',
#                 # 'AutoTheta_',
#                 # 'CrostonOptimized_',
#                 ]
# # Load results
# df_result = pd.DataFrame()
# for experiment in experiments:
#     df = pd.read_parquet(str(CURRENT_PATH.joinpath(f"{exp_folder}/{experiment}.parquet")))
#     scenario = experiment[:experiment.find('_')]
#     if scenario == 'bu':
#         df = df.rename(index = {df.index.get_level_values(1).unique()[0]:experiment}, level=1)
#     df.columns = pd.DatetimeIndex(df.columns)
#     df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
#     df = pd.concat({f"{scenario}": df}, names=['Scenario'])
#     df_result = pd.concat((df_result, df))
# df_result.columns = df_result.columns.map(pd.to_datetime)

# # Add sales bucket groupings to target values
# sales_buckets = pd.DataFrame(index = targets.loc['products', df_seed.columns].mean(1).index)
# weekly_mean = 7 * targets.loc['products', df_seed.columns].mean(1)
# sales_buckets['bucket'] = (weekly_mean <= 1) * 1
# sales_buckets['bucket'] += ((weekly_mean > 1) * (weekly_mean <= 10.0)) * 2
# sales_buckets['bucket'] += ((weekly_mean > 10) * (weekly_mean <= 100.0)) * 3
# sales_buckets['bucket'] += ((weekly_mean > 100) * (weekly_mean <= 500.0)) * 4
# sales_buckets['bucket'] += ((weekly_mean > 500)) * 5


# for scenario in scenarios:
#     for seed in seeds:
#         try:
#             df_seed = df_result.loc[(scenario, seed, slice(None), 'products')]
#             # error_seed = calc_level_method_error(forecasts_methods=df_seed, actuals=targets, metric=metric)
#             # sq_error = ((df_seed - targets.loc['products', df_seed.columns])**2).stack()
#             sq_error = ((df_seed - targets.loc['products', df_seed.columns])**2)
#             sq_error = sq_error.reset_index('Method').join(sales_buckets).reset_index('Value').set_index(['Value', 'Method', 'bucket']).stack()
#             sq_error.index.names = ['Value', 'Method', 'bucket', time_index]
#             msq_error = sq_error.groupby(['Method', 'bucket']).mean()
#             bucket_error = np.sqrt(msq_error).unstack(1)
#             error_seed = pd.concat({f'{seed}': bucket_error}, names=['Seed'])
#             error_seed = pd.concat({f"{scenario}": error_seed}, names=['Scenario'])
#             error = pd.concat((error, error_seed))
#         except:
#             pass

# error = error.groupby(["Method"]).mean()
# error /= error.loc['bu_objmse_evalmse']