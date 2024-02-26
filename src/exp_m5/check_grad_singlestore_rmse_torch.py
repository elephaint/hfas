#%% Check with autograd
import numpy as np
import pandas as pd
import torch
from hierts.reconciliation import hierarchy_temporal, hierarchy_cross_sectional
from helper_functions import read_m5, get_aggregations
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
    error_agg = (Sc @ (error @ St.T))
    denominator_c = n_levels_c * torch.sum(Sc, axis=1, keepdim=True)
    denominator_t = n_levels_t * torch.sum(St, axis=1, keepdim=True)
    loss = torch.sum(0.5 * error_agg**2 / (denominator_c @ denominator_t.T))
    
    return loss

def hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    denominator_c = 1 / (n_levels_c * torch.sum(Sc, axis=1, keepdim=True))
    denominator_t = 1 / (n_levels_t * torch.sum(St, axis=1, keepdim=True))
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    denominator = denominator_c @ denominator_t.T
    gradient_agg = Sc @ (error @ St.T) * denominator
    gradient = ((Sc.T @ gradient_agg) @ St)
    hessian = ((Sc.T @ denominator) @ St)

    return gradient, hessian

def hierarchical_obj_se_gradonly(yhat_bottom, y_bottom, Sc, St, denominator):
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    gradient_agg = Sc @ (error @ St.T) * denominator
    gradient = ((Sc.T @ gradient_agg) @ St)

    return gradient
#%% Check grad
Sc = df_Sc.sparse.to_dense().values
St = df_St.sparse.to_dense().values
Sc = torch.from_numpy(Sc).double().cuda()
St = torch.from_numpy(St).double().cuda()

y_bottom = torch.from_numpy(df_target.T.values).double().cuda()
y = (Sc @ y_bottom @ St.T).cuda()
n_levels_c = df_Sc.index.get_level_values('Aggregation').nunique()
n_levels_t = df_St.index.get_level_values('Aggregation').nunique()
yhat_bottom = torch.rand(Sc.shape[1], St.shape[1]).double().cuda()
yhat_bottom.requires_grad = True

loss = hierarchical_eval_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
loss.backward()
gradient_bottom_autograd = yhat_bottom.grad
gradient_bottom_analytical, hessian_bottom_analytical = hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
assert torch.allclose(gradient_bottom_autograd, gradient_bottom_analytical)
#%% Check diagonal hessian
eps = 1e-9
n_hessian_checks = 10000
yhat_bottom.requires_grad = False
diag_hessian_finite = torch.zeros(n_hessian_checks, dtype=torch.float64).cuda()
diag_hessian_exact = torch.zeros(n_hessian_checks, dtype=torch.float64).cuda()
series = torch.randint(low=0, high=gradient_bottom_autograd.shape[0], size=(n_hessian_checks, )).cuda()
timesteps = torch.randint(low=0, high=gradient_bottom_autograd.shape[1], size=(n_hessian_checks, )).cuda()
epsilon = torch.zeros_like(yhat_bottom)
yhat_bottom_upper = yhat_bottom.clone()
yhat_bottom_lower = yhat_bottom.clone()
# Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
denominator_c = 1 / (n_levels_c * torch.sum(Sc, axis=1, keepdim=True))
denominator_t = 1 / (n_levels_t * torch.sum(St, axis=1, keepdim=True))
denominator = denominator_c @ denominator_t.T

for k in range(n_hessian_checks):
    i = series[k]
    j = timesteps[k]
    epsilon[i, j] = eps
    gradient_upper = hierarchical_obj_se_gradonly(yhat_bottom + epsilon, y_bottom, Sc.to_sparse(), St.to_sparse(), denominator)
    gradient_lower = hierarchical_obj_se_gradonly(yhat_bottom - epsilon, y_bottom, Sc.to_sparse(), St.to_sparse(), denominator)
    diag_hessian_finite[k] = ((gradient_upper[i, j] - gradient_lower[i, j]) / (2 * eps))
    diag_hessian_exact[k] = hessian_bottom_analytical[i, j]
    epsilon[i, j] = 0.0

assert torch.allclose(diag_hessian_finite, diag_hessian_exact, atol=1e-5)