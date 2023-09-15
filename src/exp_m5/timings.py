#%% Import packages
import numpy as np
import pandas as pd
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
#%%
store_id = 0
learning_rate = 0.05
# store_level = True
# exp_folder = f"exp1_storeid={store_id}/lr{learning_rate}"
store_level = False
exp_folder = f"exp2_allstores/lr{learning_rate}"
experiments = [ 
                'globalall_objse_evalmse', 
                'bu_objmse_evalmse_dense',
                'bu_objmse_evalmse',
                # 'bu_objmse_evalhmse', 
                # 'bu_objmse_evalhmse_dense', 
                # 'bu_objtweedie_evalmse', 
                # 'bu_objtweedie_evalhmse',
                # 'bu_objtweedie_evaltweedie',
                'bu_objhse_evalhmse_dense', 
                'bu_objhse_evalhmse', 
                # 'bu_objhse_evalmse',
                # 'bu_objhse_evalhmse_withtemp',
                # 'bu_objhse_evalhmse_withtemponly',
                # 'bu_objrhse_evalhmse',
                'sepagg_objse_evalmse',
                # 'Naive',
                # 'SeasonalNaive',
                # 'AutoETS',
                # 'AutoARIMA',
                # 'AutoTheta',
                # 'CrostonOptimized',
                ]
# Load results
df_result = pd.DataFrame()
for experiment in experiments:
    df = pd.read_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{experiment}_timings.csv")), index_col=[0])
    df = pd.concat({f"{experiment}": df.mean()}, names=['Experiment']).unstack(1)
    df_result = pd.concat((df_result, df))

df_result.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/timings.csv")))
