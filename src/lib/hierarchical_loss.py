#%% Import packages
import numpy as np
from scipy.sparse import issparse, csc_matrix, vstack, eye
#%% Hierarchical loss functions
class HierarchicalLoss(object):
    def __init__(self, df_Sc, df_St):
        # Create Sc
        if hasattr(df_Sc, 'sparse'):
            Sc = csc_matrix(df_Sc.sparse.to_coo())
            n_levels_c = Sc.sum() // Sc.shape[1]
            denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
        else:
            Sc = df_Sc.values
            n_levels_c = Sc.sum() // Sc.shape[1]
            denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
        # Create St
        if hasattr(df_St, 'sparse'):
            St = csc_matrix(df_St.sparse.to_coo().T)
            n_levels_t = St.sum() // St.shape[0]
            denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
        else:
            St = df_St.values.T
            n_levels_t = St.sum() // St.shape[0]
            denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))
        
        # Assign attributes
        self.Sc = Sc
        self.St = St
        self.denominator = denominator_c @ denominator_t
        self.hessian = (Sc.T @ self.denominator @ St.T).T.reshape(-1)
    
    def objective(self, preds, train_data):
        # Bottom ground-truth and predictions, flattened
        y_bottom_flat = train_data.get_label()
        yhat_bottom_flat = preds.astype(y_bottom_flat.dtype)
        # Bottom ground-truth and predictions, reshaped
        yhat_bottom = yhat_bottom_flat.reshape(-1, self.Sc.shape[1]).T
        y_bottom = y_bottom_flat.reshape(-1, self.Sc.shape[1]).T
        # Compute gradients for all aggregations
        error = (yhat_bottom - y_bottom)
        gradient_agg = (self.Sc @ error @ self.St) * self.denominator
        # Convert gradients back to bottom-level series
        gradient = (self.Sc.T @ gradient_agg @ self.St.T).T.reshape(-1)

        return gradient, self.hessian
    
    def metric(self, preds, eval_data):
        # Bottom ground-truth and predictions, flattened
        y_bottom_flat = eval_data.get_label()
        yhat_bottom_flat = preds.astype(y_bottom_flat.dtype)
        # Bottom ground-truth and predictions, reshaped
        yhat_bottom = yhat_bottom_flat.reshape(-1, self.Sc.shape[1]).T
        y_bottom = y_bottom_flat.reshape(-1, self.Sc.shape[1]).T
        # Compute predictions for all aggregations
        y = (self.Sc @ y_bottom @ self.St)
        yhat = (self.Sc @ yhat_bottom @ self.St)
        loss = np.sum(0.5 * np.square(y - yhat) * self.denominator)
        
        return 'hierarchical_eval_hmse', np.sum(loss) / len(preds) , False

# Lightgbm objective function wrapper
def hierarchical_obj_se(preds, train_data, df_Sc, df_St):
    # Bottom ground-truth
    y_bottom_flat = train_data.get_label()
    # Create Sc
    if hasattr(df_Sc, 'sparse'):
        Sc = csc_matrix(df_Sc.sparse.to_coo())
        n_levels_c = Sc.sum() // Sc.shape[1]
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
    else:
        Sc = df_Sc.values
        n_levels_c = Sc.sum() // Sc.shape[1]
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
    # Create St
    min_date = y_bottom_flat.index.min()
    max_date = y_bottom_flat.index.max()
    df_St_fold = df_St.loc[:, min_date:max_date]
    if hasattr(df_St, 'sparse'):
        St = csc_matrix(df_St_fold.sparse.to_coo().T)
        n_levels_t = St.sum() // St.shape[0]
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
    else:
        St = df_St.values.T
        n_levels_t = St.sum() // St.shape[0]
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))
    # Compute predictions for all aggregations
    yhat_bottom = preds.astype(y_bottom_flat.dtype).reshape(-1, Sc.shape[1]).T
    y_bottom = y_bottom_flat.reshape(-1, Sc.shape[1]).T
    # Compute gradients for all aggregations
    error = (yhat_bottom - y_bottom)
    denominator = denominator_c @ denominator_t
    gradient_agg = (Sc @ error @ St) * denominator
    # Convert gradients back to bottom-level series
    gradient = (Sc.T @ gradient_agg @ St.T).T.reshape(-1)
    hessian = (Sc.T @ denominator @ St.T).T.reshape(-1)

    return gradient, hessian

# Lightgbm evaluation function wrapper
def hierarchical_eval_mse(preds, eval_data, df_Sc, df_St):
    # Bottom ground-truth
    y_bottom_flat = eval_data.get_label()
    # Create Sc
    if hasattr(df_Sc, 'sparse'):
        Sc = csc_matrix(df_Sc.sparse.to_coo())
        n_levels_c = Sc.sum() // Sc.shape[1]
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1)).A
    else:
        Sc = df_Sc.values
        n_levels_c = Sc.sum() // Sc.shape[1]
        denominator_c = 1 / (n_levels_c * np.sum(Sc, axis=1, keepdims=True))
    # Create St
    min_date = y_bottom_flat.index.min()
    max_date = y_bottom_flat.index.max()
    df_St_fold = df_St.loc[:, min_date:max_date]
    if hasattr(df_St, 'sparse'):
        St = csc_matrix(df_St_fold.sparse.to_coo().T)
        n_levels_t = St.sum() // St.shape[0]
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0), 1)).A
    else:
        St = df_St.values.T
        n_levels_t = St.sum() // St.shape[0]
        denominator_t = 1 / (n_levels_t * np.maximum(np.sum(St, axis=0, keepdims=True), 1))
    # Compute predictions for all aggregations
    yhat_bottom = preds.astype(y_bottom_flat.dtype).reshape(-1, Sc.shape[1]).T
    y_bottom = y_bottom_flat.reshape(-1, Sc.shape[1]).T
    # Compute predictions for all aggregations
    denominator = denominator_c @ denominator_t
    y = (Sc @ y_bottom @ St)
    yhat = (Sc @ yhat_bottom @ St)
    loss = np.sum(0.5 * np.square(y - yhat) * denominator)
    
    return 'hierarchical_eval_hmse', np.sum(loss) / len(preds) , False

def hierarchical_obj_se_random(preds, train_data):
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
