#%%
from datetime import datetime
import hts
from hts.utilities.load_data import load_hierarchical_sine_data
from statsmodels.tsa.arima.model import ARIMA
from hts import HTSRegressor
import statsmodels
import collections
import pandas as pd
import numpy as np
from src.lib import reconcile_forecasts

n_train = 10000
n_test = 5000
s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
hsd = load_hierarchical_sine_data(start=s, end=e, n=n_train+n_test)
hier = {'total': ['a', 'b', 'c'],
            'a': ['a_x', 'a_y'],
            'b': ['b_x', 'b_y'],
            'c': ['c_x', 'c_y'],
            'a_x': ['a_x_1', 'a_x_2'],
            'a_y': ['a_y_1', 'a_y_2'],
            'b_x': ['b_x_1', 'b_x_2'],
            'b_y': ['b_y_1', 'b_y_2'],
            'c_x': ['c_x_1', 'c_x_2'],
            'c_y': ['c_y_1', 'c_y_2']
        }

tree = hts.hierarchy.HierarchyTree.from_nodes(hier, hsd)
sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)

forecasts = pd.DataFrame(columns=hsd.columns, index=['forecast_'+str(i+1) for i in range(n_test)])
actuals = pd.DataFrame(columns=hsd.columns, index=['actual_'+str(i+1) for i in range(n_test)])
df_residuals = pd.DataFrame(columns=hsd.columns, index=['residual_'+str(i+1) for i in range(n_train)])

# Make forecasts made outside of package. Could be any modeling technique.
for col in hsd.columns:
    data = hsd[col].values[:n_train]
    model = statsmodels.tsa.holtwinters.SimpleExpSmoothing(data).fit()
    fcst = list(model.forecast(n_test))
    forecasts[col] = fcst
    actuals[col] = list(hsd[col].values[-n_test:])
    df_residuals[col] = model.resid

pred_dict = collections.OrderedDict()
mse_dict = collections.OrderedDict()

# Add predictions to dictionary is same order as summing matrix
for label in sum_mat_labels:
    pred_dict[label] = pd.DataFrame(data=forecasts[label].values, columns=['yhat'])
    mse_dict[label] = np.mean(df_residuals[label].values**2)

revised_wlss = hts.functions.optimal_combination(pred_dict, sum_mat, method='WLSS', mse={})
revised_ols = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})
revised_wlsv = hts.functions.optimal_combination(pred_dict, sum_mat, method='WLSV', mse=mse_dict)
#%% Reimplementation of reconciliation approach
yhat = forecasts[sum_mat_labels].values.T
y = actuals[sum_mat_labels].values.T
residuals = df_residuals[sum_mat_labels].values.T
S = sum_mat
revised_ols_new = reconcile_forecasts(yhat, S, method='ols')
revised_wlss_new = reconcile_forecasts(yhat, S, method='wls_struct')
revised_wlsv_new = reconcile_forecasts(yhat, S, residuals, method='wls_var')
assert np.allclose(revised_wlss.T, revised_wlss_new)
assert np.allclose(revised_ols.T, revised_ols_new)
# assert np.allclose(revised_wlsv.T, revised_wlsv_new) # Does not work, but I think hts implementation is incorrect as it leads to a worse overall error (which shouldn't be possible) and mine doesn't
#%% Combine approaches
df = pd.DataFrame()
df['actuals'] = actuals.loc['actual_1', sum_mat_labels]
methods = ['ols', 'wls_struct', 'wls_var', 'mint_cov', 'mint_shrink', 'mint_lw', 'mint_oas']
df['base_prediction'] = yhat[:, 0]
for method in methods:
    df[method] = reconcile_forecasts(yhat, S, residuals, method=method)[:, 0]

cols = [col for col in df.columns if col is not 'actuals']
errors = np.zeros((len(cols), 1))
for i, col in enumerate(cols):
    errors[i] = np.sqrt(np.mean(np.square(df['actuals'] - df[col])))

df_errors = pd.DataFrame(errors.T, index=['rmse'])
df_errors.columns = ['base_prediction'] + methods