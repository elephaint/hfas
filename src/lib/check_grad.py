#%% Check with autograd
import autograd.numpy as anp 
from autograd import grad
import numpy as np
from scipy.sparse import issparse, csc_matrix
import pandas as pd
from hierts.reconciliation import apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse, hierarchy_temporal, hierarchy_cross_sectional
#%% Read data
df = pd.read_parquet('src/exp_m5/data/m5_dataset_products.parquet', 
                    columns = ['sales', 'date', 'state_id_enc', 'store_id_enc', 'cat_id_enc', 
                                'dept_id_enc', 'item_id_enc', 'snap_CA', 'snap_TX', 'snap_WI',
                                 'event_type_1_enc', 'event_type_2_enc', 'weeks_on_sale', 'sell_price'])
df = df[(df['date'] <= '2016-05-22') & (df['date'] >= '2013-01-01') & (df['weeks_on_sale'] > 0)]
df = df.sort_values(by=['store_id_enc', 'item_id_enc', 'date']).reset_index(drop=True)
#%% Set cross-sectional aggregations
aggregations = [['state_id_enc'],
                ['store_id_enc'],
                ['cat_id_enc'],
                ['dept_id_enc'],
                ['state_id_enc', 'cat_id_enc'],
                ['state_id_enc', 'dept_id_enc'],
                ['store_id_enc', 'cat_id_enc'],
                ['store_id_enc', 'dept_id_enc'],
                ['item_id_enc'],
                ['item_id_enc', 'state_id_enc']]
target = 'sales'
time_index = 'date'
# Calculate cross-sectional hierarchy matrix
df_Sc = hierarchy_cross_sectional(df, aggregations, sparse=True)
#%% Temporal hierarchy
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.isocalendar().year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
temporal_aggregations = [['year'],
                         ['year', 'month'],
                         ['year', 'week']]
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
#%% Create target
aggregation_cols = list(set([col for cols in aggregations for col in cols]))
bottom_timeseries = pd.DataFrame(index=df_Sc.columns.str.split(pat='-',expand=True).set_names(aggregation_cols)).reset_index()
bottom_timeseries[aggregation_cols] = bottom_timeseries[aggregation_cols].astype('int')
bottom_timeseries['bottom_timeseries'] = bottom_timeseries[aggregation_cols].astype(str).agg('-'.join, axis=1)
bottom_timeseries['bottom_timeseries'] = pd.Categorical(bottom_timeseries['bottom_timeseries'], df_Sc.columns)
df[aggregation_cols] = df[aggregation_cols].astype('int')
df = df.merge(bottom_timeseries, how='left', left_on=aggregation_cols, right_on=aggregation_cols)
df_target = df.set_index([time_index, 'bottom_timeseries'])[target].unstack(1, fill_value=0)
#%% Define loss functions and evaluate gradient and hessian
def hierarchical_eval_se(yhat_bottom, y, S, n_levels):
    # Compute predictions for all aggregations
    yhat = (S @ yhat_bottom.reshape(-1, S.shape[1]).T)
    loss = anp.sum(0.5 * anp.square(y - yhat) / (n_levels * anp.sum(S, axis=1, keepdims=True)))
    
    return  anp.sum(loss)

def hierarchical_eval_se_new(yhat_bottom, y, Sc, St, n_levels_c, n_levels_t):
    # Compute predictions for all aggregations
    yhat = (Sc @ yhat_bottom.reshape(-1, Sc.shape[1]).T @ St)
    denominator_c = n_levels_c * anp.sum(Sc, axis=1, keepdims=True)
    denominator_t = n_levels_t * anp.sum(St, axis=0, keepdims=True)
    loss = anp.sum(0.5 * anp.square(y - yhat) / (denominator_c @ denominator_t))
    
    return  anp.sum(loss)

def hierarchical_obj_se2(yhat_bottom, y, S, n_levels):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    if issparse(S):
        denominator = 1 / (n_levels * np.sum(S, axis=1)).A
        hessian_step = np.asarray(np.sum(S.T.multiply(denominator.T), axis=1)).T
    else:
        denominator = 1 / (n_levels * np.sum(S, axis=1, keepdims=True))
        hessian_step = np.sum(denominator * S, axis = 0, keepdims=True).T
    # Compute predictions for all aggregations
    yhat_bottom_reshaped = yhat_bottom.astype(S.dtype).reshape(-1, S.shape[1]).T
    yhat = (S @ yhat_bottom_reshaped)
    gradient_agg = (yhat - y) * denominator
    gradient = (gradient_agg.T @ S).reshape(-1)
    hessian = hessian_step.repeat(gradient_agg.shape[1], axis=0).reshape(-1)

    return gradient, hessian

def hierarchical_obj_se_new(yhat_bottom, y, Sc, St, n_levels_c, n_levels_t):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    if issparse(Sc):
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        denominator_t = 1 / (n_levels_t * np.sum(St, axis=0)).A
    else:
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
        denominator_t = 1 / (n_levels_t * np.sum(St, axis=0, keepdims=True))
    # Compute predictions for all aggregations
    yhat_bottom_reshaped = yhat_bottom.astype(Sc.dtype).reshape(-1, Sc.shape[1]).T
    yhat = (Sc @ yhat_bottom_reshaped @ St)
    denominator = denominator_c @ denominator_t
    gradient_agg = (yhat - y) * denominator
    gradient = (Sc.T @ gradient_agg @ St.T).T.reshape(-1)
    hessian = (Sc.T @ denominator @ St.T).T.reshape(-1)

    return gradient, hessian
#%% Check grad new
rng = np.random.default_rng(seed=0)
# S = df_S.sparse.to_dense().values.astype('float64')
# Sc = df_Sc.sparse.to_coo()
Sc = df_Sc.values.astype('float64')
# St = df_St.T.sparse.to_coo()
St = df_St.values.astype('float64').T

y = (Sc @ df_target.T.values @ St).astype('float64')
n_levels_c = df_Sc.index.get_level_values('Aggregation').nunique()
n_levels_t = df_St.index.get_level_values('Aggregation').nunique()

yhat_bottom = np.random.rand(Sc.shape[1] * St.shape[0])

grad_hierarchical_se = grad(hierarchical_eval_se_new)
gradient = grad_hierarchical_se(yhat_bottom, y, Sc, St, n_levels_c, n_levels_t)
gradient_exact, hessian_exact = hierarchical_obj_se_new(yhat_bottom, y, Sc, St, n_levels_c, n_levels_t)
assert np.allclose(gradient, gradient_exact)
# auto_hessian = np.zeros(gradient.shape[0])
# eps = 1e-9
# S = df_S.sparse.to_coo()
# for i in range(gradient.shape[0]):
#     epsilon = np.zeros(gradient.shape[0])
#     epsilon[i] = eps
#     gradient_upper, _ = hierarchical_obj_se2(yhat_bottom  + epsilon, y, S, n_levels)
#     gradient_lower, _ = hierarchical_obj_se2(yhat_bottom  - epsilon, y, S, n_levels)
#     auto_hessian[i] = (gradient_upper[i] - gradient_lower[i]) / (2 * eps)
#     if i == 10:
#         break
# assert np.allclose(hessian_exact, auto_hessian)


#%%
S = np.array([[1, 1],[1, 0],[0, 1]]).astype('float64')
# S = np.array([[1, 0],[0, 1]]).astype('float64')
y_bottom = np.random.rand(S.shape[1], 2)
y = (S @ y_bottom).astype('float64')
yhat_bottom = np.random.rand(y.shape[1] * S.shape[1])
n_levels = S.sum(1).max()
# n_levels = 1

grad_hierarchical_se = grad(hierarchical_eval_se)
gradient = grad_hierarchical_se(yhat_bottom, y, S, n_levels)
gradient_exact, hessian_exact = hierarchical_obj_se2(yhat_bottom, y, S, n_levels)
assert np.allclose(gradient, gradient_exact)
auto_hessian = np.zeros(gradient.shape[0])
eps = 1e-9
for i in range(gradient.shape[0]):
    epsilon = np.zeros(gradient.shape[0])
    epsilon[i] = eps
    gradient_upper, _ = hierarchical_obj_se2(yhat_bottom  + epsilon, y, S, n_levels)
    gradient_lower, _ = hierarchical_obj_se2(yhat_bottom  - epsilon, y, S, n_levels)
    auto_hessian[i] = (gradient_upper[i] - gradient_lower[i]) / (2 * eps)
assert np.allclose(hessian_exact, auto_hessian)
#%% Sparse grad - code check for bol.com
from scipy.sparse import csc_array
import numpy as np
mask = csc_array((np.zeros(6), ([0, 0, 1, 1, 30489, 30489], [0, 1, 1236, 1237, 0, 1])), shape=(30490, 1238))
#eps_mask = csc_array((np.full(7, 1e-12), ([0, 0, 1, 1, 30489, 30489, 30489], [0, 1, 1236, 1237, 0, 1, 2])), shape=(30490, 1238), dtype=np.float32)

hessian_step = csc_array(np.sum(S.T.multiply(denominator.T), axis=1).T)
hessian_full = csc_array(hessian_step.todense().repeat(1238, axis=0))

hessian_sparse = csc_array(mask.T * hessian_step)
hessian_sparse2 = csc_array(mask.T * hessian_full)

#%%

yhat_bottom_dense = np.random.rand(df_target.shape[0], df_target.shape[1])
yhat_bottom_dense *= np.random.binomial(size=yhat_bottom_dense.shape, n=1, p=0.2)
yhat_bottom = csc_array(yhat_bottom_dense.T)
# yhat_dense = S @ yhat_bottom_dense.T

def _hierarchical_obj_se(preds, train_data, S, denominator, y, mask, date_placed_codes, \
                        global_id_codes, n_date_placed, n_global_id):
    # Compute predictions for all aggregations
    # predictions = np.maximum(preds.astype(S.dtype), 1e-6)
    predictions = preds.astype(S.dtype)
    yhat_bottom = csc_array((predictions, (global_id_codes, date_placed_codes)), \
                    shape=(n_global_id, n_date_placed), dtype=np.float32)
    eps_mask = csc_array((np.full(len(predictions), 1e-6, dtype=np.float32), (global_id_codes, date_placed_codes)), \
                    shape=(n_global_id, n_date_placed))     
    yhat = (S @ yhat_bottom)
    # Compute gradients for all aggregations
    gradient_agg = csc_array((yhat - y) * denominator)
    # Convert gradients back to bottom-level series
    gradient_full = csc_array(gradient_agg.T @ S)
    # Apply mask to only keep entries that are also in preds
    gradient_sparse = csc_array(mask.T * gradient_full)
    # Output gradient and hessian
    gradient = gradient_sparse.data
    
    # Calculate hessian
    hessian_step = csc_array(np.sum(S.T.multiply(denominator.T), axis=1).T)
    # Apply mask to only keep entries that are also in preds
    hessian_sparse = csc_array(mask.T * hessian_step)
    hessian = hessian_sparse.data

    return gradient, hessian