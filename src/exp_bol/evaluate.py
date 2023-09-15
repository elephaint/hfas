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
sales_buckets =['0-1', '2-10', '11-100', '101-500', '501+']

error = 'mae'
baseline = 'objtweedie_evall2'
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
labels = ["Hierarchical Loss", "Squared Loss", "Tweedie Loss"]
leg = fig.legend(handles, labels, loc = 'lower right', ncol=1)
# fig.legend(handles, labels=labels, loc='center')
leg.set_bbox_to_anchor((0.90, 0.1))
leg.get_frame().set_linewidth(0.0)
# sns.move_legend(ax, "center", frameon=False)
fig.tight_layout()
fig.delaxes(axes[2, 1])
