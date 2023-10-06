#%%
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
#%%
CURRENT_PATH = Path(__file__).parent
filename = "errors_['horizon', 'sales_bucket'].csv"
df = pd.read_csv(str(CURRENT_PATH.joinpath(filename)), index_col=0)
df = df[df["Scenario"] != 'baseline']
df = df.set_index(['sales_bucket','horizon',  'Scenario'])
series_to_plot = ['2022_baseline', '2022_objl2_evall2_logtransformed', '2022_objhse_evalhmse_logrerun']
# series_to_plot = ['2022_baseline', '2022_objl2_evall2_bol', '2022_objhse_evalhmse_m5']

df = df.loc[(slice(None), slice(None), series_to_plot)]

sales_buckets =['0-1', '2-10', '11-100', '101-500', '501+']

error = 'mae'
baseline = '2022_baseline'
fig, axes = plt.subplots(3, 2)
for i, sales_bucket in enumerate(sales_buckets):
    ax = fig.axes[i]
    df_current = df.loc[sales_buckets[i], error]
    df_current = df_current.unstack(0)
    df_current /= df_current.loc[baseline]
    sns.lineplot(ax = ax, data=df_current.T)
    ax.tick_params(labelsize=12)
    ax.get_legend().set_visible(False)
    # ax.spines['top'].set_color('white') 
    # ax.spines['right'].set_color('white')
    if i % 2 == 0:
        ax.set_ylabel(ylabel = f'{error}', fontsize=12)
    ax.set_title(f"{sales_bucket}")

# ax =  fig.axes[-1]
# sns.lineplot(ax = ax, data=df_current.T)
# ax.get_legend().set_visible(False)
handles, labels = ax.get_legend_handles_labels()
labels = ["Tweedie Loss", "Squared Loss", "Hierarchical Loss", "Hierarchical Loss2"]
leg = fig.legend(handles, labels, loc = 'lower right', ncol=1)
# fig.legend(handles, labels=labels, loc='center')
leg.set_bbox_to_anchor((0.90, 0.1))
leg.get_frame().set_linewidth(0.0)
# sns.move_legend(ax, "center", frameon=False)
fig.tight_layout()
fig.delaxes(axes[2, 1])
#%% Level errors
CURRENT_PATH = Path(__file__).parent
filenames = ["errors_['horizon'].csv",
            "errors_['horizon', 'product_group'].csv",
            "errors_['horizon', 'seasonality_group'].csv",
            # "errors_['horizon', 'seasonality_group', 'product_group'].csv"
            ]
filenames_plot = ["Products", "Product group", "Seasonality group"]

series_to_plot = ['2022_baseline', '2022_objl2_evall2_logtransformed', '2022_objhse_evalhmse_logrerun']
# series_to_plot = ['2022_baseline', '2022_objl2_evall2_bol', '2022_objhse_evalhmse_m5']
errors = ['mae_level', 'rmse_level']
errors_plot_string = ['MAE', 'RMSE']
baseline = '2022_baseline'

n_rows = len(filenames)
n_cols = len(errors)
# fig, axes = plt.subplots(n_rows, n_cols)
fig = plt.figure(constrained_layout=True)
# fig.set_size_inches(6, 6)
# fig.suptitle('Figure title')
subfigs = fig.subfigures(nrows=n_rows, ncols=1)
for i in range(n_rows):
    filename = filenames[i]
    df = pd.read_csv(str(CURRENT_PATH.joinpath(filename)), index_col=0)
    df = df.sort_values(by = ['Scenario', 'horizon']).set_index(['Scenario',  'horizon'])
    df = df[['rmse_level', 'mae_level']]
    df = df.loc[series_to_plot]
    subfig = subfigs[i]
    subfig.suptitle(f'{filenames_plot[i]}')
    axs = subfig.subplots(nrows=1, ncols=n_cols)
    for j in range(n_cols):
        ax = axs[j]
        # ax = fig.axes[i * n_cols + j]
        df_current = df.loc[:, errors[j]]
        df_current /= df_current.loc[baseline]
        sns.lineplot(ax = ax, data=df_current.unstack(0))
        ax.tick_params(labelsize=12)
        # ax.get_legend().remove()
        ax.get_legend().set_visible(False)
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.set_ylabel(ylabel = f'{errors_plot_string[j]}', fontsize=12)
        ax.set_xlabel(xlabel=None)
        if i > 2:
            ax.set_xlabel(xlabel = 'horizon', fontsize=12)
        print(f"Mean error: {df_current.loc['2022_objhse_evalhmse_logrerun'].mean()}")

handles, labels = ax.get_legend_handles_labels()
labels = ["Tweedie Loss", "Squared Loss", "Hierarchical Loss"]
leg = fig.legend(handles, labels, loc = 'lower center', ncol=3)
leg.set_bbox_to_anchor((0.5, -0.025))
leg.get_frame().set_linewidth(0.0)
# fig.tight_layout()
