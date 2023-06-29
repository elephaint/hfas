def _hierarchical_obj_se(preds, train_data, S, denominator, y, mask, date_placed_codes, \
                        global_id_codes, n_date_placed, n_global_id):
    # Compute predictions for all aggregations
    predictions = np.maximum(preds.astype(S.dtype), 1e-6)
    yhat_bottom = csc_array((predictions, (global_id_codes, date_placed_codes)), \
                    shape=(n_global_id, n_date_placed), dtype=np.float32)    
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
    hessian_step = np.asarray(np.sum(S.T.multiply(denominator.T), axis=1)).T
    hessian_full = csc_array(hessian_step.repeat(gradient_agg.shape[1], axis=0))
    # Apply mask to only keep entries that are also in preds
    hessian_sparse = csc_array(mask.T * hessian_full)
    hessian = hessian_sparse.data

    return gradient, hessian


# Shape gradient_full:(1210, 3049)
# Shape gradient_agg: (3060, 1210)
# Hessian step shape: (1, 3049)
# Hessian shape: (1, 3689290)

def _hierarchical_obj_se(preds, train_data, S, denominator, y, mask, date_placed_codes, \
                        global_id_codes, n_date_placed, n_global_id):
    # Compute predictions for all aggregations
    # predictions = np.maximum(preds.astype(S.dtype), 1e-6)
    predictions = preds.astype(S.dtype)
    yhat_bottom = csc_array((predictions, (global_id_codes, date_placed_codes)), \
                    shape=(n_global_id, n_date_placed), dtype=np.float32)
    # eps_mask = csc_array((np.full(len(predictions), 1e-6, dtype=np.float32), (global_id_codes, date_placed_codes)), \
    #                 shape=(n_global_id, n_date_placed))     
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