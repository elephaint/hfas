#%% Check with autograd
import autograd.numpy as anp 
from autograd import grad
import numpy as np
from scipy.sparse import issparse, csc_matrix
import pandas as pd
from hierts.reconciliation import calc_summing_matrix, apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse
#%% Read data
df = pd.read_parquet('src/exp_m5/data/m5_dataset_products.parquet', 
                    columns = ['sales', 'date', 'state_id_enc', 'store_id_enc', 'cat_id_enc', 
                                'dept_id_enc', 'item_id_enc', 'snap_CA', 'snap_TX', 'snap_WI',
                                 'event_type_1_enc', 'event_type_2_enc', 'weeks_on_sale', 'sell_price'])
df = df[(df['date'] <= '22-05-2016') & (df['date'] >= '01-01-2013') & (df['weeks_on_sale'] > 0)]
df = df.sort_values(by=['store_id_enc', 'item_id_enc', 'date']).reset_index(drop=True)
#%% Set aggregations and target
aggregation_cols = ['state_id_enc', 'store_id_enc', 'cat_id_enc', 'dept_id_enc', 'item_id_enc']
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
# Calculate summing matrix
df_S = calc_summing_matrix(df[[time_index] + aggregation_cols], aggregation_cols, aggregations, sparse=True)
bottom_timeseries = pd.DataFrame(index=df_S.columns.str.split(pat='-',expand=True).set_names(aggregation_cols)).reset_index()
bottom_timeseries[aggregation_cols] = bottom_timeseries[aggregation_cols].astype('int')
bottom_timeseries['bottom_timeseries'] = bottom_timeseries[aggregation_cols].astype(str).agg('-'.join, axis=1)
bottom_timeseries['bottom_timeseries'] = pd.Categorical(bottom_timeseries['bottom_timeseries'], df_S.columns)
df[aggregation_cols] = df[aggregation_cols].astype('int')
df = df.merge(bottom_timeseries, how='left', left_on=aggregation_cols, right_on=aggregation_cols)
df_target = df.set_index([time_index, 'bottom_timeseries'])[target].unstack(1, fill_value=0)
#%% Define loss functions and evaluate gradient and hessian
def hierarchical_eval_se(yhat, y, S, n_levels):
    # Compute predictions for all aggregations
    yhat = (S @ yhat.reshape(-1, S.shape[1]).T)
    loss = anp.sum(0.5 * anp.square(y - yhat) / (n_levels * anp.sum(S, axis=1, keepdims=True)))
    
    return  anp.sum(loss)

def hierarchical_obj_se2(yhat, y, S, n_levels):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    if issparse(S):
        denominator = (n_levels * np.sum(S, axis=1)).A
    else:
        denominator = (n_levels * np.sum(S, axis=1, keepdims=True))
    # Compute predictions for all aggregations
    yhat_bottom = yhat.astype(S.dtype).reshape(-1, S.shape[1]).T
    yhat = (S @ yhat_bottom)
    gradient_agg = (yhat - y) / denominator
    gradient = (gradient_agg.T @ S).reshape(-1)
    hessian = np.ones_like(gradient)

    return gradient, hessian

rng = np.random.default_rng(seed=0)
S = df_S.sparse.to_dense().values.astype('float64')
y = (S @ df_target.T.values).astype('float64')
yhat = np.random.rand(y.shape[1] * S.shape[1])
n_levels = df_S.index.get_level_values('Aggregation').nunique()
grad_hierarchical_se = grad(hierarchical_eval_se)
gradient = grad_hierarchical_se(yhat, y, S, n_levels)
gradient_exact, hessian_exact = hierarchical_obj_se2(yhat, y, S, n_levels)
assert np.allclose(gradient, gradient_exact)
epsilon = 1e-6
auto_hessian = (grad_hierarchical_se(yhat + epsilon, y, S, n_levels) - grad_hierarchical_se(yhat - epsilon, y, S, n_levels)) / (2 * epsilon)
assert np.allclose(hessian_exact, auto_hessian)