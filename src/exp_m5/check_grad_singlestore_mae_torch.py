#%% Check with autograd
import autograd.numpy as anp 
from autograd import grad
import numpy as np
from scipy.sparse import issparse, csc_matrix
import pandas as pd
import torch
from hierts.reconciliation import hierarchy_temporal, hierarchy_cross_sectional
from helper_functions import read_m5, get_aggregations, create_forecast_set
#%% Read data
store_level = True
# store_level = False
store_id = 0
cross_sectional_aggregations, temporal_aggregations = get_aggregations(store_level)
time_index = 'date'
target = 'sales'
name_bottom_timeseries = 'products'
end_train = pd.to_datetime('2016-04-24')
start_test = pd.to_datetime('2016-04-25')
df = read_m5(store_level=store_level, store_id=store_id)
# Add columns for temporal hierarchies
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# Calculate cross-sectional and temporal hierarchy summing matrices
df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
# Create forecast set
aggregation_cols = list(dict.fromkeys([col for cols in cross_sectional_aggregations for col in cols]))
df = df.drop(columns = ['week', 'year', 'month', 'day'])
#%% Create target
bottom_timeseries = pd.DataFrame(index=df_Sc.columns.str.split(pat='-',expand=True).set_names(aggregation_cols)).reset_index()
bottom_timeseries[aggregation_cols] = bottom_timeseries[aggregation_cols].astype('int')
bottom_timeseries['bottom_timeseries'] = bottom_timeseries[aggregation_cols].astype(str).agg('-'.join, axis=1)
bottom_timeseries['bottom_timeseries'] = pd.Categorical(bottom_timeseries['bottom_timeseries'], df_Sc.columns)
df[aggregation_cols] = df[aggregation_cols].astype('int')
df = df.merge(bottom_timeseries, how='left', left_on=aggregation_cols, right_on=aggregation_cols)
df_target = df.set_index([time_index, 'bottom_timeseries'])[target].unstack(1, fill_value=0)
#%% Define loss functions and evaluate gradient and hessian
def hierarchical_eval_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    error_agg = torch.abs(Sc @ (error @ St.T))
    denominator_c = n_levels_c * torch.sum(Sc, axis=1, keepdim=True)
    denominator_t = n_levels_t * torch.sum(St, axis=1, keepdim=True)
    loss = torch.sum(error_agg / (denominator_c @ denominator_t.T))
    
    return loss

def hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    if issparse(Sc):
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        denominator_t = 1 / (n_levels_t * np.sum(St, axis=1)).A
    else:
        denominator_c = 1 / (n_levels_c * torch.sum(Sc, axis=1, keepdim=True))
        denominator_t = 1 / (n_levels_t * torch.sum(St, axis=1, keepdim=True))
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    denominator = denominator_c @ denominator_t.T
    gradient_agg = torch.sign(Sc @ (error @ St.T)) * denominator
    gradient = ((Sc.T @ gradient_agg) @ St)
    hessian = ((Sc.T @ denominator) @ St).T

    return gradient, hessian
#%% Check grad new
Sc = df_Sc.sparse.to_dense().values
St = df_St.sparse.to_dense().values
Sc = torch.from_numpy(Sc)
St = torch.from_numpy(St)

y_bottom = torch.from_numpy(df_target.T.values).float()
y = (Sc @ y_bottom @ St.T)
n_levels_c = df_Sc.index.get_level_values('Aggregation').nunique()
n_levels_t = df_St.index.get_level_values('Aggregation').nunique()
yhat_bottom = torch.rand(Sc.shape[1], St.shape[1])
yhat_bottom.requires_grad = True

loss = hierarchical_eval_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
loss.backward()
gradient_exact, hessian_exact = hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
assert torch.allclose(yhat_bottom.grad, gradient_exact, atol=1e-6)