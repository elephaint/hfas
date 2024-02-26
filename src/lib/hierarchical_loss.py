#%% Import packages
import numpy as np
from scipy.sparse import csc_matrix, vstack, eye, issparse
# Use sparse-dot-mkl for speed-up if available
# Install: conda install -c conda-forge sparse_dot_mkl
try:
    import sparse_dot_mkl
    dot_product = sparse_dot_mkl.dot_product_mkl
except ImportError:
    def dot_product(x, y, dense=False, **kwargs):
        z = x @ y
        return z.A if dense and issparse(z) else z 
#%% Hierarchical loss functions
def prepare_HierarchicalLoss(n_bottom_timeseries, n_bottom_timesteps, 
                             df_Sc=None, df_St=None):
    # Create Sc
    if df_Sc is None:
        Sc = eye(n_bottom_timeseries, dtype=np.float32)
        denominator_c = np.full((n_bottom_timeseries, 1), fill_value=1, dtype=np.float32)
    else:
        assert n_bottom_timeseries == df_Sc.shape[1]
        if hasattr(df_Sc, 'sparse'):
            Sc = df_Sc.sparse.to_coo().tocsc().astype('float32')
            n_levels_c = Sc.sum() // Sc.shape[1]
            denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        else:
            Sc = df_Sc.values.astype('float32')
            n_levels_c = Sc.sum() // Sc.shape[1]
            denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
    # Create St
    if df_St is None:
        St = eye(n_bottom_timesteps, dtype=np.float32)
        denominator_t = np.full((1, n_bottom_timesteps), fill_value=1, dtype=np.float32)
    else:
        assert n_bottom_timesteps == df_St.shape[1]
        if hasattr(df_St, 'sparse'):
            St = df_St.sparse.to_coo().tocsc().T.astype('float32')
            n_levels_t = St.sum() // St.shape[0]
            denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
        else:
            St = df_St.values.T.astype('float32')
            n_levels_t = St.sum() // St.shape[0]
            denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))
        
    # Compute denominator and hessian
    denominator = denominator_c @ denominator_t
    hessian = ((Sc.T @ denominator) @ St.T).T.reshape(-1)
    if hasattr(df_Sc, 'sparse'):
        Scd = Sc.multiply(denominator_c).tocsc()
        Std = St.multiply(denominator_t).tocsc()
    else:
        Scd = Sc * denominator_c
        Std = St * denominator_t.T

    return hessian, denominator, Sc, Scd, St, Std

def prepare_RandomHierarchicalLoss(n_bottom_timeseries, n_bottom_timesteps,
                                   max_levels_random,
                                   max_categories_per_random_level, rng):
    # Create random hierarchy
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
    Sc = vstack([S_top, S_aggs, S_bottom]).tocsc()
    n_levels_c = Sc.sum() // Sc.shape[1]
    denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
    # Create St
    St = eye(n_bottom_timesteps, dtype=np.float32)
    denominator_t = np.full((1, n_bottom_timesteps), fill_value=1, dtype=np.float32)       
    # Compute denominator and hessian
    denominator = denominator_c @ denominator_t
    hessian = ((Sc.T @ denominator) @ St.T).T.reshape(-1)
    Scd = Sc.multiply(denominator_c).tocsc()
    Std = St.multiply(denominator_t).tocsc()

    return hessian, denominator, Sc, Scd, St, Std

def HierarchicalLossObjective(preds, train_data, hessian,  
                              n_bottom_timeseries, n_bottom_timesteps, 
                              Sc=None, Scd=None, St=None, Std=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, flattened
    y_bottom_flat = train_data.get_label()
    yhat_bottom_flat = preds.astype(y_bottom_flat.dtype)
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(n_bottom_timesteps, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(n_bottom_timesteps, n_bottom_timeseries).T
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

def HierarchicalLossMetric(preds, eval_data, denominator, 
                           n_bottom_timeseries, n_bottom_timesteps, Sc=None, St=None):
    assert (Sc is not None or St is not None), "Sc, St or both should be provided"
    # Bottom ground-truth and predictions, flattened
    y_bottom_flat = eval_data.get_label()
    yhat_bottom_flat = preds.astype(y_bottom_flat.dtype)
    # Bottom ground-truth and predictions, reshaped
    yhat_bottom = yhat_bottom_flat.reshape(n_bottom_timesteps, n_bottom_timeseries).T
    y_bottom = y_bottom_flat.reshape(n_bottom_timesteps, n_bottom_timeseries).T
    # Compute error for all aggregations
    error = (yhat_bottom - y_bottom)
    if Sc is None:
        error_agg = dot_product(error, St)
    elif St is None:
        error_agg = dot_product(Sc, error)
    else:
        error_agg = dot_product(Sc, dot_product(error, St))

    loss = np.sum(0.5 * np.square(error_agg) * denominator)
    
    return 'hierarchical_eval_hmse', np.sum(loss) / len(preds), False

def RandomHierarchicalLossObjective(preds, train_data, rng, n_bottom_timeseries, 
                                    max_levels_random, max_categories_per_random_level,
                                    hier_freq):
    # Draw random number
    number = rng.uniform()
    if number < (1 / hier_freq):
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
        S = vstack([S_top, S_aggs, S_bottom]).tocsc()
        # Compute predictions for all aggregations
        yhat_bottom = preds.astype(np.float32).reshape(-1, S.shape[1]).T
        y_bottom = train_data.get_label().astype(np.float32).reshape(-1, S.shape[1]).T
        # Calculate denominator
        n_bottom_timesteps = yhat_bottom.shape[1]
        denominator_c = 1 / ((n_levels_random + 2) * np.sum(S, axis=1)).A
        denominator_t = np.full((1, n_bottom_timesteps), fill_value=1, dtype=np.float32)
        denominator = dot_product(denominator_c, denominator_t)
        # Compute bottom level error
        error = (yhat_bottom - y_bottom)
        # Compute aggregated gradients and convert back to bottom-level
        gradient_agg = dot_product(S, error) * denominator
        gradient = dot_product(S.T, gradient_agg).T.reshape(-1)
        hessian = dot_product(S.T, denominator).T.reshape(-1)
    else:
        gradient = (preds - train_data.get_label())
        hessian = np.ones_like(gradient)

    return gradient, hessian