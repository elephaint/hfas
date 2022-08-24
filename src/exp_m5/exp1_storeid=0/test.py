#%%
import os
print(os.getcwd())
import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods, calc_level_method_rmse
from src.exp_m5.helper_functions import read_m5, create_forecast_set 
from src.exp_m5.experiments import exp_m5_globalall, exp_m5_sepagg, exp_m5_globalbottomup

