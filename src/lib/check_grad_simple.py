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
    error = anp.log1p(y_bottom) - anp.log1p(yhat_bottom)
    # error = y_bottom - yhat_bottom
    error_agg = (Sc @ (error @ St.T))
    denominator_c = n_levels_c * anp.sum(Sc, axis=1, keepdims=True)
    denominator_t = n_levels_t * anp.sum(St, axis=1, keepdims=True).T
    # loss = anp.sum(0.5 * anp.square(error_agg) / (denominator_c @ denominator_t))    
    loss = anp.sum(0.5 * anp.square(error_agg))    

    return  anp.sum(loss)
    
    # error = anp.log1p(y_bottom) - anp.log1p(yhat_bottom)
    # loss = anp.sum(0.5 * anp.square(error))

    # return loss

def hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t):
    # Address discrepancy in the output and workings of np.sum with sparse vs dense arrays
    # denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
    # denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=1, keepdims=True), 1))
    # denominator = denominator_c @ denominator_t.T
    # Compute predictions for all aggregations
    error =  (np.log1p(yhat_bottom) - np.log1p(y_bottom))
    factor = (1 / (1 + yhat_bottom))
    # error = yhat_bottom - y_bottom

    # Compute aggregated gradients and convert back to bottom-level
    # Scd = Sc * denominator_c
    # Std = St * denominator_t
    # gradient_agg = (Scd @ error @ Std.T)
    gradient_agg = (Sc @ error @ St.T)

    gradient = factor * ((Sc.T @ gradient_agg) @ St)
    # hessian = (1 / (1 + yhat_bottom)**2) * ((Sc.T @ denominator) @ St) + 

    hessian = (-1 / (1 + yhat_bottom)**2) * ((Sc.T @ gradient_agg) @ St) + factor * (Sc.T @ St)

    # hessian = (-1 / (1 + yhat_bottom)**2) * ((Sc.T @ denominator) @ St)

    # gradient = 1 / (1 + yhat_bottom) * (np.log1p(yhat_bottom) - np.log1p(y_bottom))
    # hessian = ((Sc.T @ denominator) @ St)

    return gradient, hessian

#%% Gradient
Sc = np.array([[1, 1],[1, 0],[0, 1]]).astype('float64')
St = np.array([[1, 1],[1, 0],[0, 1]]).astype('float64')
# Sc = np.array([[1, 0],[0, 1]]).astype('float64')
# St = np.array([[1, 0],[0, 1]]).astype('float64')
y_bottom = np.array([[1, 1.2], [1.2, 0.2]]).astype('float64')
yhat_bottom = np.array([[0.9, 0.1], [1.1, 0.25]]).astype('float64')

n_levels_c = Sc.sum() // Sc.shape[1]
n_levels_t = St.sum() // St.shape[1]

grad_hierarchical_se = grad(hierarchical_eval_se)
gradient = grad_hierarchical_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
gradient_exact, hessian_exact = hierarchical_obj_se(yhat_bottom, y_bottom, Sc, St, n_levels_c, n_levels_t)
assert np.allclose(gradient, gradient_exact)
#%% Hessian
auto_hessian = np.zeros_like(y_bottom)
eps = 1e-9
for i in range(y_bottom.shape[0]):
    for j in range(y_bottom.shape[1]):
        epsilon = np.zeros_like(y_bottom)
        epsilon[i, j] = eps
        gradient_upper = grad_hierarchical_se(yhat_bottom  + epsilon, y_bottom, Sc,  St, n_levels_c, n_levels_t)
        gradient_lower = grad_hierarchical_se(yhat_bottom  - epsilon, y_bottom, Sc,  St, n_levels_c, n_levels_t)
        auto_hessian[i, j] = ((gradient_upper - gradient_lower) / (2 * eps))[i, j]
assert np.allclose(hessian_exact, auto_hessian)