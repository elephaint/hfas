#%% Read packages
import sparse_dot_mkl
import pandas as pd
import numpy as np
import time
from hierts.reconciliation import hierarchy_temporal, hierarchy_cross_sectional
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
from helper_functions import read_m5, get_aggregations, create_forecast_set
from scipy.sparse import csr_array, csc_array
from numba import njit, prange
#%% Set experiment parameters
store_level = True
store_id = 0
cross_sectional_aggregations, temporal_aggregations = get_aggregations(store_level)
name_bottom_timeseries = 'products'
time_index = 'date'
#%% Read data
df = read_m5(store_level=store_level, store_id=store_id)
# Add columns for temporal hierarchies
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
if store_level:
    df_Sc = df_Sc.drop('item_id_enc') 
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
df = df.drop(columns = ['week', 'year', 'month', 'day'])
#%% Objective
def objective_scipy(y_bottom_flat, yhat_bottom_flat, hessian, 
                              n_bottom_timeseries, n_bottom_timesteps,
                             Sc=None, Scd=None, St=None, Std=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(-1, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(-1, n_bottom_timeseries).T
    # Compute bottom level error
    error = (yhat_bottom - y_bottom)
    # Compute aggregated gradients and convert back to bottom-level
    if Sc is None:
        gradient_agg = (error @ Std)
        gradient = (gradient_agg @ St.T).T.reshape(-1)
    elif St is None:
        gradient_agg = (Scd @ error)
        gradient = (gradient_agg.T @ Sc).reshape(-1)
    else:
        gradient_agg = (Scd @ (error @ Std))
        gradient = ((Sc.T @ gradient_agg) @ St.T).T.reshape(-1)

    return gradient, hessian

def objective_mkl(y_bottom_flat, yhat_bottom_flat, hessian,  
                              n_bottom_timeseries, n_bottom_timesteps,
                             Sc=None, Scd=None, St=None, Std=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(-1, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(-1, n_bottom_timeseries).T
    # Compute bottom level error
    error = (yhat_bottom - y_bottom)
    # Compute aggregated gradients and convert back to bottom-level
    if Sc is None:
        gradient_agg = dot_product(error, Std)
        gradient = dot_product(gradient_agg, St.T).T.reshape(-1)
    elif St is None:
        gradient_agg = dot_product(Scd, error)
        gradient = dot_product(gradient_agg.T, Sc).reshape(-1)
    else:
        gradient_agg = dot_product(Scd, dot_product(error, Std))
        gradient = dot_product(Sc.T, dot_product(gradient_agg, St.T)).T.reshape(-1)

    return gradient, hessian  

def objective_numba_csr(y_bottom_flat, yhat_bottom_flat, hessian, 
                              n_bottom_timeseries, n_bottom_timesteps,
                             Sc=None, Scd=None, St=None, Std=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(-1, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(-1, n_bottom_timeseries).T
    # Compute bottom level error
    error = (yhat_bottom - y_bottom)
    # Compute aggregated gradients and convert back to bottom-level
    if Sc is None:
        gradient_agg = semm(dot_product(error, St), denominator)
        gradient = dot_product(gradient_agg, St.T).T.reshape(-1)
    elif St is None:
        gradient_agg = csr_smdm(Scd.tocsr().indptr, Scd.tocsr().indices, error, Scd.shape[0])
        gradient = csr_smdm(Sc.T.tocsr().indptr, Sc.T.tocsr().indices, gradient_agg, Sc.T.shape[0]).T.reshape(-1)
    else:
        dot1 = csr_smdm(Scd.tocsr().indptr, Scd.tocsr().indices, error, Scd.shape[0])
        gradient_agg = csr_smdm(Std.T.tocsr().indptr, Std.T.tocsr().indices, dot1.T, Std.T.shape[0]).T
        dot2 = csr_smdm(Sc.T.tocsr().indptr, Sc.T.tocsr().indices, gradient_agg, Sc.T.shape[0])
        gradient = csr_smdm(St.tocsr().indptr, St.tocsr().indices, dot2.T, St.shape[0]).reshape(-1)

    return gradient, hessian  

def objective_numba_csc(y_bottom_flat, yhat_bottom_flat, hessian, denominator, 
                              n_bottom_timeseries, n_bottom_timesteps,
                             Sc=None, St=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(-1, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(-1, n_bottom_timeseries).T
    # Compute bottom level error
    error = (yhat_bottom - y_bottom)
    # Compute aggregated gradients and convert back to bottom-level
    if Sc is None:
        gradient_agg = semm(dot_product(error, St), denominator)
        gradient = dot_product(gradient_agg, St.T).T.reshape(-1)
    elif St is None:
        gradient_agg = csc_smdmpdm(Sc.tocsc().indptr, Sc.tocsc().indices, error, denominator)
        gradient = csc_smdm(Sc.T.tocsc().indptr, Sc.T.tocsc().indices, gradient_agg, Sc.T.shape[0]).T.reshape(-1)
    else:
        dot1 = csc_smdm(Sc.tocsc().indptr, Sc.tocsc().indices, error, Sc.shape[0])
        gradient_agg = csc_smdmpdm(St.T.tocsc().indptr, St.T.tocsc().indices, dot1.T, denominator.T).T
        dot2 = csc_smdm(Sc.T.tocsc().indptr, Sc.T.tocsc().indices, gradient_agg, Sc.T.shape[0])
        gradient = csc_smdm(St.tocsc().indptr, St.tocsc().indices, dot2.T, St.shape[0]).reshape(-1)

    return gradient, hessian  


@njit(fastmath=True, parallel=True)
def semm(A, B):
    # Single-precision Element-wise Matrix-Matrix Multiplication
    C = np.zeros((A.shape[1], A.shape[0]), dtype=A.dtype).T
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] * B[i, j]
    
    return C

@njit(fastmath=True, parallel=True)
def csr_smdmpdm(Ap, Aj, Bd, Cd):
    """This function computes the following product:
    CSR Sparse-Matrix @ Dense-Matrix Plus Dense Matrix

    Yd = (A @ Bd) * Cd

    A  [M x N] is a sparse CSR-matrix with nnz non-zero elements
        Ap[M + 1] is the row pointer of A (A.indptr)
        Aj[nnz(A)] is column indices of A (A.indices)
    Bd [N x P] is a dense float32-array
    Cd [M x P] is a dense float32-array

    Complexity is O(M/p * N * nnz) for p parallel threads

    This is a modification from csr_matvec of Scipy:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/sparse/sparsetools/csr.h
   
    """
    n_row = Cd.shape[0]
    n_col = Cd.shape[1]
    Yd = np.zeros((n_col, n_row), dtype=np.float32).T
    for j in prange(n_col):
        for i in range(n_row):
            row_start = Ap[i]
            row_end = Ap[i + 1]
            sum = 0
            for jj in range(row_start, row_end):
                sum += Bd[Aj[jj], j] * Cd[i, j]
            Yd[i, j] = sum
    return Yd

@njit(fastmath=True, parallel=True)
def csr_smdm(Ap, Aj, Bd, n_row):
    """This function computes the following product:
    CSR Sparse-Matrix @ Dense-Matrix

    Yd = (A @ Bd)

    A  [M x N] is a sparse CSR-matrix with nnz non-zero elements
        Ap[M + 1] is the row pointer of A (A.indptr)
        Aj[nnz(A)] is column indices of A (A.indices)
    Bd [N x P] is a dense float32-array
    n_row = M

    Complexity is O(M/p * N * nnz) for p parallel threads

    This is a modification from csr_matvec of Scipy:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/sparse/sparsetools/csr.h
    
    """
    n_col = Bd.shape[1]
    Yd = np.zeros((n_col, n_row), dtype=np.float32).T
    for j in prange(n_col):
        for i in range(n_row):
            row_start = Ap[i]
            row_end = Ap[i + 1]
            sum = 0
            for jj in range(row_start, row_end):
                sum += Bd[Aj[jj], j]
            Yd[i, j] = sum
    return Yd

@njit(fastmath=True, parallel=True)
def csc_smdmpdm(Ap, Ai, Bd, Cd):
    """This function computes the following product:
    CSC Sparse-Matrix @ Dense-Matrix Plus Dense Matrix

    Yd = (A @ Bd) * Cd

    A  [M x N] is a sparse CSC-matrix with nnz non-zero elements
        Ap[M + 1] is the column pointer of A (A.indptr)
        Aj[nnz(A)] is row indices of A (A.indices)
    Bd [N x P] is a dense float32-array
    Cd [M x P] is a dense float32-array

    Complexity is O(P/p * M * nnz) for p parallel threads

    This is a modification from csc_matvec of Scipy:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/sparse/sparsetools/csc.h
   
    """
    n_row = Cd.shape[0]
    n_col_out = Cd.shape[1]
    n_col_in = len(Ap) - 1
    Yd = np.zeros((n_col_out, n_row), dtype=np.float32).T
    for i in prange(n_col_out):
        for j in range(n_col_in):
            col_start = Ap[j]
            col_end = Ap[j + 1]
            for ii in range(col_start, col_end):
                index = Ai[ii]
                Yd[index, i] += Bd[j, i] * Cd[index, i]    
    return Yd

@njit(fastmath=True, parallel=True)
def csc_smdm(Ap, Ai, Bd, n_row):
    """This function computes the following product:
    CSC Sparse-Matrix @ Dense-Matrix

    Yd = (A @ Bd)

    A  [M x N] is a sparse CSC-matrix with nnz non-zero elements
        Ap[M + 1] is the column pointer of A (A.indptr)
        Aj[nnz(A)] is row indices of A (A.indices)
    Bd [N x P] is a dense float32-array
    n_row = M

    Complexity is O(P/p * M * nnz) for p parallel threads

    This is a modification from csc_matvec of Scipy:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/sparse/sparsetools/csc.h
    
    """
    n_col_out = Bd.shape[1]
    n_col_in = len(Ap) - 1
    Yd = np.zeros((n_col_out, n_row), dtype=np.float32).T
    for i in prange(n_col_out):
        for j in range(n_col_in):
            col_start = Ap[j]
            col_end = Ap[j + 1]
            for ii in range(col_start, col_end):
                index = Ai[ii]
                Yd[index, i] += Bd[j, i]    
    return Yd

#%% Time functions
rng = np.random.default_rng(seed=12345678)
dot_product = sparse_dot_mkl.dot_product_mkl
yhat_bottom_flat = rng.random((df_Sc.shape[1], df_St.shape[1]), dtype=np.float32).reshape(-1)
y_bottom_flat = rng.random((df_Sc.shape[1], df_St.shape[1]), dtype=np.float32).reshape(-1)
n_iters = 10
Sc = df_Sc.sparse.to_coo().tocsc().astype('float32')
St = df_St.sparse.to_coo().tocsc().T.astype('float32')
n_bottom_timeseries = Sc.shape[1]
n_bottom_timesteps = St.shape[0]
n_levels_c = Sc.sum() // Sc.shape[1]
denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
n_levels_t = St.sum() // St.shape[0]
denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
# denominator_t = np.full((1, n_bottom_timesteps), fill_value=1, dtype=np.float32)
denominator = denominator_c @ denominator_t
Scd = Sc.multiply(denominator_c).tocsr()
Std = St.multiply(denominator_t).tocsr()
hessian = ((Sc.T @ denominator) @ St.T).T.reshape(-1)
# hessian = (Sc.T @ denominator).T.reshape(-1)
#%%
t_mkl = np.zeros((n_iters, 1))
t_scipy = np.zeros((n_iters, 1))
t_numba_csr = np.zeros((n_iters, 1))
t_numba_csc = np.zeros((n_iters, 1))
for i in range(n_iters):
    
    start = time.perf_counter()
    gradient_scipy, hessian = objective_scipy(y_bottom_flat, yhat_bottom_flat, hessian,
                                  n_bottom_timeseries, n_bottom_timesteps,
                                  Sc, Scd, St=St, Std=Std)
    # gradient_scipy, hessian = objective_scipy(y_bottom_flat, yhat_bottom_flat, hessian,
    #                               n_bottom_timeseries, n_bottom_timesteps,
    #                               Sc, Scd, St=None, Std=None)
    end = time.perf_counter()
    t_scipy[i, 0] = (end - start) * 1000

    start = time.perf_counter()
    # gradient_mkl, hessian = objective_mkl(y_bottom_flat, yhat_bottom_flat, hessian,
    #                               n_bottom_timeseries, n_bottom_timesteps,
    #                               Sc, Scd, St=None, Std=None)
    gradient_mkl, hessian = objective_mkl(y_bottom_flat, yhat_bottom_flat, hessian,
                                  n_bottom_timeseries, n_bottom_timesteps,
                                  Sc, Scd, St=St, Std=Std)
    end = time.perf_counter()
    t_mkl[i, 0] = (end - start) * 1000

    start = time.perf_counter()
    # gradient_numba_csr, hessian = objective_numba_csr(y_bottom_flat, yhat_bottom_flat, hessian,
    #                               n_bottom_timeseries, n_bottom_timesteps,
    #                               Sc, Scd, St=None, Std=None)    
    end = time.perf_counter()
    t_numba_csr[i, 0] = (end - start) * 1000

    start = time.perf_counter()
    # gradient_numba_csc, hessian = objective_numba_csc(y_bottom_flat, yhat_bottom_flat, hessian,
    #                               denominator, n_bottom_timeseries, n_bottom_timesteps,
    #                               Sc, St=None)    
    gradient_numba_csc, hessian = objective_numba_csc(y_bottom_flat, yhat_bottom_flat, hessian,
                                  denominator, n_bottom_timeseries, n_bottom_timesteps,
                                  Sc, St=St)    
    end = time.perf_counter()
    t_numba_csc[i, 0] = (end - start) * 1000

    assert np.allclose(gradient_scipy, gradient_mkl, atol=1e-7)
    # assert np.allclose(gradient_scipy, gradient_numba_csr, atol=1e-7)
    assert np.allclose(gradient_scipy, gradient_numba_csc, atol=1e-7)
    
#%%
print(f"Time scipy    : {t_scipy.mean(0)}")
print(f"Time mkl      : {t_mkl.mean(0)}")
print(f"Time numba_csr: {t_numba_csr.mean(0)}")
print(f"Time numba_csc: {t_numba_csc.mean(0)}")