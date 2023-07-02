#%% Import packages
import numpy as np
from scipy.sparse import issparse, csc_matrix, vstack, eye
#%% Hierarchical loss functions
# Lightgbm objective function wrapper
def hierarchical_obj_se(preds, train_data, S):
    n_levels = S.sum() // S.shape[1]
    # Switch sparse/dense
    if issparse(S):
        denominator = 1 / (n_levels * np.sum(S, axis=1)).A
        hessian_step = np.asarray(np.sum(S.T.multiply(denominator.T), axis=1)).T    
    else:
        denominator = 1 / (n_levels * np.sum(S, axis=1, keepdims=True))
        hessian_step = np.sum(S.T * denominator.T, axis=1, keepdims=True).T

    # Compute predictions for all aggregations
    yhat_bottom = preds.astype(S.dtype).reshape(-1, S.shape[1]).T
    y_bottom = train_data.get_label().astype(S.dtype).reshape(-1, S.shape[1]).T
    # Compute gradients for all aggregations
    gradient_agg = (S @ (yhat_bottom - y_bottom)) * denominator
    # Convert gradients back to bottom-level series
    gradient = (gradient_agg.T @ S).reshape(-1)
    # Compute hessian
    hessian = hessian_step.repeat(gradient_agg.shape[1], axis=0).reshape(-1)

    return gradient, hessian

def hierarchical_obj_se_withtemp(preds, train_data, Sc, St):
    # Get required fields
    assert type(Sc) == type(St), 'Cross-sectional and temporal hierarchy matrices should have the same datatype'
    n_levels_c = Sc.sum() // Sc.shape[1]
    n_levels_t = St.sum() // St.shape[0]
    # Switch sparse/dense
    if issparse(Sc) and issparse(St):
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
    else:
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))
    # Compute predictions for all aggregations
    yhat_bottom = preds.astype(np.float64).reshape(-1, Sc.shape[1]).T
    y_bottom = train_data.get_label().astype(np.float64).reshape(-1, Sc.shape[1]).T
    # Compute gradients for all aggregations
    error = (yhat_bottom - y_bottom)
    denominator = denominator_c @ denominator_t
    gradient_agg = (Sc @ error @ St) * denominator
    # Convert gradients back to bottom-level series
    gradient = (Sc.T @ gradient_agg @ St.T).T.reshape(-1)
    hessian = (Sc.T @ denominator @ St.T).T.reshape(-1)

    return gradient, hessian

def hierarchical_obj_se_random(preds, train_data, S=None):
    # Get required fields
    assert 'n_bottom_timeseries' in train_data.params, 'Train data should contain parameter n_bottom_timeseries, the number of bottom timeseries in the hierarchy'   
    assert 'max_levels_random' in train_data.params, 'Train data should contain the parameter max_levels_random'
    assert 'max_categories_per_random_level' in train_data.params, 'Train data should contain the parameter max_categories_per_random_level'
    assert 'hier_freq' in train_data.params, 'Train data should contain the parameter hier_freq, the frequency of using the random hierarchical loss'
    # Draw random number
    rng = np.random.default_rng()
    number = rng.uniform()
    if number < (1 / train_data.params['hier_freq']):
        # Get data
        n_bottom_timeseries = train_data.params['n_bottom_timeseries']
        max_levels_random = np.maximum(train_data.params['max_levels_random'], 1)
        max_categories_per_random_level = np.maximum(train_data.params['max_categories_per_random_level'], 2)
        # Create random aggregations
        ones = np.ones(n_bottom_timeseries, dtype=np.float32)
        idx_range = np.arange(n_bottom_timeseries)
        n_levels_random = rng.integers(1, max_levels_random + 1)
        S_aggs_list = []
        for _ in range(n_levels_random):
            n_categories_per_level = rng.integers(2, max_categories_per_random_level + 1)
            codes = rng.integers(0, n_categories_per_level, size=(n_bottom_timeseries, ))
            S_agg = csc_matrix((ones, (codes, idx_range)))
            S_aggs_list.append(S_agg)
        S_aggs = vstack(S_aggs_list)
        # Create top and bottom level
        S_top = csc_matrix(ones, dtype=np.float32)
        S_bottom = eye(n_bottom_timeseries, dtype=np.float32)
        # Construct S: stack top, aggregations and bottom 
        S = vstack([S_top, S_aggs, S_bottom])
        # Calculate gradient and hessian
        denominator = 1 / ((n_levels_random + 2) * np.sum(S, axis=1)).A
        # Compute predictions for all aggregations
        yhat_bottom = preds.astype(S.dtype).reshape(-1, S.shape[1]).T
        y_bottom = train_data.get_label().astype(S.dtype).reshape(-1, S.shape[1]).T
        yhat = (S @ yhat_bottom)
        y = (S @ y_bottom)
        # Compute gradients for all aggregations
        gradient_agg = (yhat - y) * denominator
        # Convert gradients back to bottom-level series
        gradient = (gradient_agg.T @ S).reshape(-1)
        hessian_step = np.asarray(np.sum(S.T.multiply(denominator.T), axis=1)).T
        # hessian = hessian_step.repeat(gradient_agg.shape[1], -1).reshape(-1)
        hessian = hessian_step.repeat(gradient_agg.shape[1], axis=0).reshape(-1)
    else:
        gradient = (preds - train_data.get_label())
        hessian = np.ones_like(gradient)

    return gradient, hessian

# Lightgbm objective function wrapper
def hierarchical_eval_mse(preds, eval_data, S):
    # Get required fields
    n_levels = S.sum() // S.shape[1]
    # Switch sparse/dense
    if issparse(S):
        denominator = 1 / (n_levels * np.sum(S, axis=1)).A
        hessian_step = np.asarray(np.sum(S.T.multiply(denominator.T), axis=1)).T    
    else:
        denominator = 1 / (n_levels * np.sum(S, axis=1, keepdims=True))
        hessian_step = np.sum(S.T * denominator.T, axis=1, keepdims=True).T
    # Compute predictions for all aggregations
    y = (S @ eval_data.get_label().astype(S.dtype).reshape(-1, S.shape[1]).T)
    yhat = (S @ preds.astype(S.dtype).reshape(-1, S.shape[1]).T)
    loss = np.sum(0.5 * np.square(y - yhat) * denominator)
    
    return 'hierarchical_eval_hmse', np.sum(loss) / len(preds) , False

# Lightgbm objective function wrapper
def hierarchical_eval_mse_withtemp(preds, eval_data, Sc, St):
    # Get required fields
    assert type(Sc) == type(St), 'Cross-sectional and temporal hierarchy matrices should have the same datatype'
    # Get levels per hierarchy
    n_levels_c = Sc.sum() // Sc.shape[1]
    n_levels_t = St.sum() // St.shape[0]
    # Calculate denominators. NB this can be done offline once!
    if issparse(Sc):
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
    else:
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))    
    # Compute predictions for all aggregations
    denominator = denominator_c @ denominator_t
    y = (Sc @ eval_data.get_label().astype(Sc.dtype).reshape(-1, Sc.shape[1]).T @ St)
    yhat = (Sc @ preds.astype(Sc.dtype).reshape(-1, Sc.shape[1]).T @ St)
    loss = np.sum(0.5 * np.square(y - yhat) * denominator)
    
    return 'hierarchical_eval_hmse_withtemp', np.sum(loss) / len(preds) , False