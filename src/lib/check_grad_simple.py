#%% Check with autograd
import autograd.numpy as anp 
from autograd import grad
import numpy as np
from scipy.sparse import issparse, csc_matrix
import pandas as pd
from hierts.reconciliation import apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse, hierarchy_temporal, hierarchy_cross_sectional
#%% Define loss functions and evaluate gradient and hessian
def hierarchical_eval_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    error_agg = (Sc @ (error @ St.T))
    denominator_c = n_levels_c * anp.sum(Sc, axis=1, keepdims=True)
    denominator_t = n_levels_t * anp.sum(St, axis=1, keepdims=True).T
    loss = anp.sum(0.5 * anp.square(error_agg) / (denominator_c @ denominator_t))
    
    return  anp.sum(loss)

def hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
    denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=1, keepdims=True), 1))
    denominator = denominator_c @ denominator_t.T
    # Compute predictions for all aggregations
    error = yhat_bottom - y_bottom
    # Compute aggregated gradients and convert back to bottom-level
    Scd = Sc * denominator_c
    Std = St * denominator_t
    gradient_agg = Scd @ error @ Std.T
    gradient = ((Sc.T @ gradient_agg) @ St)
    hessian = ((Sc.T @ denominator) @ St)

    return gradient, hessian

#%%
# Sc = np.array([[1, 1],[1, 0],[0, 1]]).astype('float64')
# St = np.array([[1, 1],[1, 0],[0, 1]]).astype('float64')
Sc = np.array([[1, 0],[0, 1]]).astype('float64')
St = np.array([[1, 0],[0, 1]]).astype('float64')
y_bottom = np.array([[1, 0], [1.2, 0.2]]).astype('float64')
yhat_bottom = np.array([[0.9, 0.1], [1.1, 0.25]]).astype('float64')

n_levels_c = Sc.sum() // Sc.shape[1]
n_levels_t = St.sum() // St.shape[1]

grad_hierarchical_se = grad(hierarchical_eval_se)
gradient = grad_hierarchical_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
gradient_exact, hessian_exact = hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
assert np.allclose(gradient, gradient_exact)
# auto_hessian = np.zeros(gradient.shape[0])
# eps = 1e-9
# for i in range(gradient.shape[0]):
#     epsilon = np.zeros(gradient.shape[0])
#     epsilon[i] = eps
#     gradient_upper, _ = hierarchical_obj_se2(yhat_bottom  + epsilon, y, S, n_levels)
#     gradient_lower, _ = hierarchical_obj_se2(yhat_bottom  - epsilon, y, S, n_levels)
#     auto_hessian[i] = (gradient_upper[i] - gradient_lower[i]) / (2 * eps)
# assert np.allclose(hessian_exact, auto_hessian)