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
    # model = ARIMA(data).fit()
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

# Put reconciled forecasts in nice DataFrame form
# revised_forecasts = pd.DataFrame(data=revised[0:,0:],
#                                     index=forecasts.index,
#                                     columns=sum_mat_labels)

#%% Reimplmentation of reconciliation approach
import numpy as np 
from sklearn.covariance import empirical_covariance, ledoit_wolf

def reconcile_forecasts(yhat, S, residuals=None, method='ols'):
    """Optimal reconciliation of hierarchical forecasts using various approaches.
    
        Based on approaches from:
    
        Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). 
        Optimal forecast reconciliation for hierarchical and grouped time series through 
        trace minimization. Journal of the American Statistical Association, 114(526), 804-819.

        :param yhat: forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]
        :type yhat: numpy.ndarray
        :param S: summing matrix detailing the hierarchical tree of size [n_timeseries x n_bottom_timeseries]
        :type S: numpy.ndarray
        :param residuals: residuals (i.e. forecast errors) for each time series for a set of historical timesteps of 
        size [n_timeseries x n_timesteps_residuals]. Only used in 'wls_var', 'mint_cov' and 'mint_shrink'.
        :type residuals: numpy.ndarray, optional
        :param method: reconciliation method, defaults to 'ols'. Options are: 'ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink'
        :type method: str, optional

        :return: ytilde, reconciled forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]
        :rtype: numpy.ndarray

    """
    n_timeseries = S.shape[0]
    n_bottom_timeseries = S.shape[1]
    assert yhat.shape[0] == n_timeseries, "Forecasts and summing matrix S do not contain the same amount of time series"
    St = np.transpose(S)
    # Select correct diagonal matrix according to reconciliation method
    if method == 'wls_var':
        # Weighted least squares using variance scaling
        assert residuals is not None, "wls_var requires a residual matrix"
        assert residuals.shape[0] == n_timeseries, "Residuals and summing matrix S do not contain the same amount of time series"
        Winv = np.diag(1 / np.mean(residuals**2, 1))
    elif method == 'wls_struct':
        # Weighted least squares using structural scaling
        unit_vector = np.full(n_bottom_timeseries, n_bottom_timeseries**(-0.5))
        Winv = np.diag(1 / (S @ unit_vector))
    elif method == 'mint_cov':
        # Trace minimization using the empirical error covariance matrix
        assert residuals is not None, "mint_cov requires a residual matrix"
        assert residuals.shape[0] == n_timeseries, "Residuals and summing matrix S do not contain the same amount of time series"
        covariance_estimate = empirical_covariance(residuals.T)
        Winv = np.linalg.inv(covariance_estimate)
    elif method == 'mint_shrink':
        # Trace minimization using the shrunk covariance matrix based on Ledoit-Wolf method
        assert residuals is not None, "mint_shrink requires a residual matrix"
        assert residuals.shape[0] == n_timeseries, "Residuals and summing matrix S do not contain the same amount of time series"
        covariance_estimate, _ = ledoit_wolf(residuals.T)
        Winv = np.linalg.inv(covariance_estimate)
    else:
        # Weighted (ordinary) least squares, default option
        Winv = np.eye(n_timeseries)
        
    # Compute reconciled forecasts
    ytilde = (S @ np.linalg.inv(St @ Winv @ S) @ St @ Winv) @ yhat

    return ytilde

yhat = forecasts[sum_mat_labels].values.T
y = actuals[sum_mat_labels].values.T
residuals = df_residuals[sum_mat_labels].values.T
S = sum_mat
revised_ols_new = reconcile_forecasts(yhat, S, method='ols')
assert np.allclose(revised_ols.T, revised_ols_new)
revised_wlss_new = reconcile_forecasts(yhat, S, method='wls_struct')
assert np.allclose(revised_wlss.T, revised_wlss_new)
revised_wlsv_new = reconcile_forecasts(yhat, S, residuals, method='wls_var')
# assert np.allclose(revised_wlsv.T, revised_wlsv_new) # Does not work, but I think hts implementation is incorrect
revised_mint_cov_new = reconcile_forecasts(yhat, S, residuals, method='mint_cov')
revised_mint_shrink_new = reconcile_forecasts(yhat, S, residuals, method='mint_shrink')
#%% Combine approaches
df = pd.DataFrame()
df['actuals'] = actuals.loc['actual_1', sum_mat_labels]
df['base_predictions'] = forecasts.loc['forecast_1', sum_mat_labels]
df['ols'] = revised_ols_new[:, 0]
df['wls_struct'] = revised_wlss_new[:, 0]
df['wls_var'] = revised_wlsv_new[:, 0]
df['mint_cov'] = revised_mint_cov_new[:, 0]
df['mint_shrink'] = revised_mint_shrink_new[:, 0]

cols = [col for col in df.columns if col is not 'actuals']
errors = np.zeros(6)
for i, col in enumerate(cols):
    errors[i] = np.sqrt(np.mean(np.square(df['actuals'] - df[col])))