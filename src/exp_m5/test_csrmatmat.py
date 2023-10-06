#%%
from numba import njit, prange
import numpy as np
from scipy.sparse import random, csr_array, hstack, vstack, csc_array
from scipy import stats
from numpy.random import default_rng
from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl
from hierts.reconciliation import hierarchy_temporal, hierarchy_cross_sectional, hierarchy_cross_sectional_array
from helper_functions import read_m5, get_aggregations, create_forecast_set
from numba import uint32, float32
import numba
numba.set_num_threads(8)
#%% Matmat parallel
@njit(['(int32, int32, int32[::1], int32[::1], int32[::1], int32[::1])'], locals = {'i': uint32, 'jj_start': uint32, 'jj_end': uint32, 'j': uint32, 'kk_start': uint32, 'kk_end': uint32, 'k': uint32}, fastmath=True, parallel=True, inline="always")
def csr_matmat_nnz(n_row, n_col, Ap, Aj, Bp, Bj):

    nnz_rows = np.zeros(n_row + 1, dtype=np.int32)
    for i in prange(n_row):
        mask = np.full(n_col, -1, dtype=np.int32)
        jj_start = Ap[i]
        jj_end   = Ap[i + 1]
        for jj in range(jj_start, jj_end):
            j = Aj[jj]
            
            kk_start = Bp[j]
            kk_end   = Bp[j + 1]
            for kk in range(kk_start, kk_end):
                k = Bj[kk]
                
                if mask[k] != i:
                    mask[k] = i
                    nnz_rows[i + 1] += 1

    return nnz_rows

@njit(['(int32, int32, int32[::1], int32[::1], float32[::1], int32[::1], int32[::1], float32[::1], int32[::1], int32[::1], float32[::1])'], locals = {'i': uint32, 'nnz': uint32, 'head': uint32, 'length': uint32, 'jj_start': uint32, 'jj_end': uint32, 'j': uint32, 'v': uint32, 'kk_start': uint32, 'kk_end': uint32, 'k': uint32, 'jj': uint32, 'temp': uint32}, fastmath=True, parallel=True, inline="always")
def csr_matmat_inline(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx):
    for i in prange(n_row):
        next = np.full(n_col, -1, dtype=np.int32)
        sums = np.zeros(n_col, dtype=np.float32)
        nnz = Cp[i]

        head   = -2
        length =  0

        jj_start = Ap[i]
        jj_end   = Ap[i+1]

        for jj in range(jj_start, jj_end):
            j = Aj[jj]
            v = Ax[jj]

            kk_start = Bp[j]
            kk_end   = Bp[j + 1]

            for kk in range(kk_start, kk_end):
                k = Bj[kk]
                sums[k] += v * Bx[kk]

                if next[k] == -1:
                    next[k] = head
                    head  = k
                    length += 1
        
        for jj in range(length):
            if sums[head] != 0:
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1

            temp = head
            head = next[head]

            next[temp] = -1 
            sums[temp] =  0
    
    return Cp, Cj, Cx


@njit(['(int32, int32, int32[::1], int32[::1], float32[::1], int32[::1], int32[::1], float32[::1])'], fastmath=True, parallel=True)
def csr_matmat_optim(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx):

    nnz_rows = csr_matmat_nnz(n_row, n_col, Ap, Aj, Bp, Bj)

    # Create Cp, Cj, Cx
    Cp = np.cumsum(nnz_rows)
    Cj = np.zeros(Cp[-1], dtype=np.int32)
    Cx = np.ones(Cp[-1], dtype=np.float32)  

    # Compute dot_product
    Cp, Cj, Cx = csr_matmat_inline(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx)

    return Cp, Cj, Cx

def dot_product(A, B):
    # Get pointers
    Ap = A.indptr
    Aj = A.indices
    Ax = A.data
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    n_row = A.shape[0]
    n_col = B.shape[1]
   
    # Compute A @ B
    Cp, Cj, Cx = csr_matmat_optim(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx)
    # Create sparse return array
    C = csr_array((Cx, Cj, Cp), shape=(n_row, n_col)) 
    
    return C        
#%% M5 data
store_level = False
store_id = 0
cross_sectional_aggregations, temporal_aggregations = get_aggregations(store_level)
name_bottom_timeseries = 'products'
time_index = 'date'
df = read_m5(store_level=store_level, store_id=store_id)
# Add columns for temporal hierarchies
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
# Sca = hierarchy_cross_sectional_array(df, cross_sectional_aggregations, sparse=True)
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
df = df.drop(columns = ['week', 'year', 'month', 'day'])
Sc = df_Sc.sparse.to_coo().tocsr().astype('float32')
A = Sc
B = Sc.T.tocsr()
#%% Random
rng = np.random.default_rng()
rvs = stats.poisson(25, loc=10).rvs
n_rowsA, n_colsA, n_colsB = 42840, 30490, 42840
A = random(n_rowsA, n_colsA, density=0.00028, random_state=rng, data_rvs=rvs, dtype=np.float32).tocsr()
B = random(n_colsA, n_colsB, density=0.00028, random_state=rng, data_rvs=rvs, dtype=np.float32).tocsr()

#%% Dot product timings
C1 = A @ B
C2 = dot_product_mkl(A, B)
C3 = dot_product(A, B)
# assert np.allclose(C1.todense(), C2.todense())
# assert np.allclose(C1.todense(), C3.todense())
#%%
%timeit C1 = A @ B
%timeit C2 = dot_product_mkl(A, B)
%timeit C3 = dot_product(A, B)