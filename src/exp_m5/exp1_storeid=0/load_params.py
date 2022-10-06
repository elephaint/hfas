#%%
import joblib
params = joblib.load('src/exp_m5/exp1_storeid=0/bu_objtweedie_evalhmse_best_params.params')
#%%
params_new = joblib.load('src/exp_m5/exp1_storeid=0/globalall_objse_evalmse_best_params.params')
