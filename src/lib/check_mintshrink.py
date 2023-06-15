#%%
import numpy as np
from numba import njit, prange
#%%
# @njit(parallel=True, fastmath=True, error_model='numpy', nogil=True)
# @njit(fastmath=True, error_model='numpy', nogil=True)
@njit(fastmath=True, error_model='numpy', nogil=True, parallel=True)
def shrunk_covariance_schaferstrimmer(residuals, residuals_mean, residuals_std):
    """Shrink empirical covariance according to the following method:
        Schäfer, Juliane, and Korbinian Strimmer. 
        ‘A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and 
        Implications for Functional Genomics’. Statistical Applications in 
        Genetics and Molecular Biology 4, no. 1 (14 January 2005). 
        https://doi.org/10.2202/1544-6115.1175.

    :meta private:
    """
    n_timeseries = residuals.shape[0]
    n_samples = residuals.shape[1]
    # We need the empirical covariance, the off-diagonal sum of the variance of 
    # the empirical correlation matrix and the off-diagonal sum of the squared 
    # empirical correlation matrix.
    emp_cov = np.zeros((n_timeseries, n_timeseries), dtype=np.float32)
    sum_var_emp_corr = np.float32(0)
    sum_sq_emp_corr = np.float32(-n_timeseries)
    factor_emp_corr = n_samples / (n_samples - 1)
    factor_var_emp_cor = n_samples / (n_samples - 1)**3
    for i in prange(n_timeseries):
        # Calculate standardized residuals
        X_i = (residuals[i] - residuals_mean[i]) 
        Xs_i = X_i / residuals_std[i]
        Xs_i_mean = np.mean(Xs_i)
        for j in range(n_timeseries):
            # Calculate standardized residuals
            X_j = (residuals[j] - residuals_mean[j]) 
            Xs_j = X_j / residuals_std[j]
            Xs_j_mean = np.mean(Xs_j)
            # Empirical covariance
            emp_cov[i, j] = factor_emp_corr * np.mean(X_i * X_j)
            # Sum off-diagonal variance of empirical correlation
            w = (Xs_i - Xs_i_mean) * (Xs_j - Xs_j_mean)
            w_mean = np.mean(w)
            sum_var_emp_corr += (i != j) * factor_var_emp_cor * np.sum(np.square(w - w_mean))
            # Sum squared empirical correlation (off-diagonal correction made by initializing 
            # with -n_timeseries, so (i != j) not necessary here)
            sum_sq_emp_corr += np.square(factor_emp_corr * w_mean)

    # Calculate shrinkage intensity 
    shrinkage = sum_var_emp_corr / sum_sq_emp_corr
    # Calculate shrunk covariance estimate
    emp_cov_diag = np.diag(emp_cov)
    W = (1 - shrinkage) * emp_cov
    # Fill diagonal with original empirical covariance diagonal
    np.fill_diagonal(W, emp_cov_diag)

    return W
#%%
# n_timeseries = 42840
# n_samples = 1210
n_timeseries = 428
n_samples = 12
rng = np.random.default_rng(0)
residuals = rng.normal(size=(n_timeseries, n_samples))
residuals_mean = np.mean(residuals, axis=1)
residuals_std = np.std(residuals, axis=1)
W = shrunk_covariance_schaferstrimmer(residuals, residuals_mean, residuals_std)
