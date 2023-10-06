#%%
import numpy as np
from hierts.reconciliation import hierarchy_temporal, hierarchy_cross_sectional, hierarchy_cross_sectional_array, hierarchy_temporal_array
from helper_functions import read_m5, get_aggregations, create_forecast_set
import polars as pl
import pandas as pd
from scipy.sparse import csr_array, coo_matrix, vstack
import time
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
dfp = df.copy()

# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
Sca = hierarchy_cross_sectional_array(df, cross_sectional_aggregations, sparse=True)
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
Sta = hierarchy_temporal_array(df, time_index, temporal_aggregations, sparse=True)

# df = df.drop(columns = ['week', 'year', 'month', 'day'])
Sc = df_Sc.sparse.to_coo().tocsr().astype('float32')
St = df_St.sparse.to_coo().tocsr().astype('float32')


assert np.allclose(Sc.data, Sca.data)
assert np.allclose(Sc.indptr, Sca.indptr)
assert np.allclose(Sc.indices, Sca.indices)
assert np.allclose(St.data, Sta.data)
assert np.allclose(St.indptr, Sta.indptr)
assert np.allclose(St.indices, Sta.indices)

#%% Pandas
df = dfp
start = time.perf_counter()
# Check whether aggregations are in the df
aggregation_cols_in_aggregations = list(dict.fromkeys([col for cols in aggregations for col in cols]))
for col in aggregation_cols_in_aggregations:
    assert col in df.columns, f"Column {col} in aggregations not present in df"
# Find the unique aggregation columns from the given set of aggregations
levels = df[aggregation_cols_in_aggregations].drop_duplicates()
name_bottom_timeseries = 'bottom_timeseries'
levels[name_bottom_timeseries] = levels[aggregation_cols_in_aggregations].astype(str).agg('-'.join, axis=1)
levels = levels.sort_values(by=name_bottom_timeseries).reset_index(drop=True)
n_bottom_timeseries = len(levels)
aggregations_total = aggregations + [[name_bottom_timeseries]]
# Check if we have not introduced redundant columns. If so, remove that column.
for col, n_uniques in levels.nunique().items():
    if col != name_bottom_timeseries and n_uniques == n_bottom_timeseries:
        levels = levels.drop(columns=col)
        aggregations_total.remove([col])
# Create summing matrix for all aggregation levels
ones = np.ones(n_bottom_timeseries, dtype=np.float32)
idx_range = np.arange(n_bottom_timeseries)
# Create summing matrix (=row vector) for top level (=total) series
S = csr_array(ones)
for aggregation in aggregations_total:
    agg = pd.Categorical(levels[aggregation].astype(str).agg('-'.join, axis=1))
    S_agg_sp = csr_array(coo_matrix((ones, (agg.codes, idx_range))))
    S = vstack((S, S_agg_sp), format="csr")

end = time.perf_counter()
Spd = S
print(f"Time pandas: {end-start:.4f}s")
#%% Polars
df = pl.from_pandas(dfp)
start = time.perf_counter()
aggregations = cross_sectional_aggregations
# Check whether aggregations are in the df
aggregation_cols_in_aggregations = list(dict.fromkeys([col for cols in aggregations for col in cols]))
for col in aggregation_cols_in_aggregations:
    assert col in df.columns, f"Column {col} in aggregations not present in df"
# Find the unique aggregation columns from the given set of aggregations
levels = df[aggregation_cols_in_aggregations].unique()
name_bottom_timeseries = 'bottom_timeseries'
levels = levels.cast(pl.Utf8)
levels = levels.with_columns(pl.concat_str(pl.col(aggregation_cols_in_aggregations), separator='-').alias(name_bottom_timeseries))
levels = levels.sort(by=name_bottom_timeseries)
n_bottom_timeseries = len(levels)
aggregations_total = aggregations + [[name_bottom_timeseries]]
# Check if we have not introduced redundant columns. If so, remove that column.
for col in levels.columns:
    n_uniques = levels[col].n_unique()
    if col != name_bottom_timeseries and n_uniques == n_bottom_timeseries:
        levels = levels.drop(columns=col)
        aggregations_total.remove([col])
# Create summing matrix for all aggregation levels
ones = np.ones(n_bottom_timeseries, dtype=np.float32)
idx_range = np.arange(n_bottom_timeseries)
# Create summing matrix (=row vector) for top level (=total) series
S = csr_array(ones)
for aggregation in aggregations_total:
    agg = levels.select(pl.concat_str(pl.col(aggregation), separator='-').alias('agg'))
    codes = pd.Categorical(agg['agg'].to_pandas()).codes
    S_agg_sp = csr_array(coo_matrix((ones, (codes, idx_range))))
    S = vstack((S, S_agg_sp), format="csr")

end = time.perf_counter()
Spl = S
print(f"Time polars: {end-start:.4f}s")
assert np.allclose(Spd.data, Spl.data)
assert np.allclose(Spd.indptr, Spl.indptr)
assert np.allclose(Spd.indices, Spl.indices)
#%% Pandas
from pandas.api.types import is_datetime64_any_dtype as is_datetime
time_column = time_index
df = dfp
aggregations = temporal_aggregations
sparse = True
start = time.perf_counter()
assert time_column in df.columns, "The time_column is not a column in the dataframe"
assert is_datetime(df[time_column]), "The time_column should be a datetime64-dtype. Use `pd.to_datetime` to convert objects to the correct datetime format."
# Check whether aggregations are in the df
aggregation_cols_in_aggregations = list(dict.fromkeys([col for cols in aggregations for col in cols]))
for col in aggregation_cols_in_aggregations:
    assert col in df.columns, f"Column {col} in aggregations not present in df"
# Find the unique aggregation columns from the given set of aggregations
levels = df[aggregation_cols_in_aggregations + [time_column]].drop_duplicates()
levels = levels.sort_values(by=time_column).reset_index(drop=True)
n_bottom_timestamps = len(levels)
aggregations_total = aggregations + [[time_column]]
# Create summing matrix for all aggregation levels
ones = np.ones(n_bottom_timestamps, dtype=np.float32)
idx_range = np.arange(n_bottom_timestamps)
S = csr_array(np.zeros(n_bottom_timestamps, dtype=np.float32))
for aggregation in aggregations_total:
    agg = pd.Categorical(levels[aggregation].astype(str).agg('-'.join, axis=1))
    S_agg_sp = csr_array(coo_matrix((ones, (agg.codes, idx_range))))
    S = vstack((S, S_agg_sp), format="csr")

# Stack all summing matrices: aggregations, bottom
end = time.perf_counter()
Spd = S
print(f"Time pandas: {end-start:.4f}s")
#%% Polars
from pandas.api.types import is_datetime64_any_dtype as is_datetime
time_column = time_index
df = pl.from_pandas(dfp)
aggregations = temporal_aggregations
sparse = True
start = time.perf_counter()
assert time_column in df.columns, "The time_column is not a column in the dataframe"
assert df[time_column].dtype == pl.Datetime, "The time_column should be a datetime64-dtype. "
# Check whether aggregations are in the df
aggregation_cols_in_aggregations = list(dict.fromkeys([col for cols in aggregations for col in cols]))
for col in aggregation_cols_in_aggregations:
    assert col in df.columns, f"Column {col} in aggregations not present in df"
# Find the unique aggregation columns from the given set of aggregations
levels = df[aggregation_cols_in_aggregations + [time_column]].unique()
levels = levels.sort(by=time_column)
n_bottom_timestamps = len(levels)
aggregations_total = aggregations + [[time_column]]
# Create summing matrix for all aggregation levels
ones = np.ones(n_bottom_timestamps, dtype=np.float32)
idx_range = np.arange(n_bottom_timestamps)
S = csr_array(np.zeros(n_bottom_timestamps, dtype=np.float32))
for aggregation in aggregations_total:
    agg = levels.select(pl.concat_str(pl.col(aggregation), separator='-').alias('agg'))
    codes = pd.Categorical(agg['agg'].to_pandas()).codes
    S_agg_sp = csr_array(coo_matrix((ones, (codes, idx_range))))
    S = vstack((S, S_agg_sp), format="csr")

# Stack all summing matrices: aggregations, bottom
end = time.perf_counter()
Spl = S
print(f"Time polars: {end-start:.4f}s")
assert np.allclose(Spd.data, Spl.data)
assert np.allclose(Spd.indptr, Spl.indptr)
assert np.allclose(Spd.indices, Spl.indices)
