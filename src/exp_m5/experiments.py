import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from src.lib import hierarchical_obj_se, hierarchical_obj_se_withtemp, hierarchical_eval_mse, hierarchical_eval_mse_withtemp, hierarchical_obj_se_random
from hierts.reconciliation import aggregate_bottom_up_forecasts
from scipy.sparse import csc_matrix
from lightgbm import early_stopping, log_evaluation
from functools import partial
#%% Hyperparameter tuning helper functions
# Cross-validation iterator
def cv_iterator(X_val, time_index, n_splits=6, n_days_test=28, return_dates=False):
    indices = np.arange(X_val.shape[0])
    date_max = X_val.index.get_level_values(time_index).max()
    indices_list, dates_list = [], []
    for i in range(n_splits, 0, -1):
        date_end_train = date_max - pd.Timedelta(i * n_days_test, 'd')
        date_start_train = date_end_train - pd.Timedelta(2 * 365, 'd')
        date_start_test = date_end_train + pd.Timedelta(1, 'd')
        date_end_test = date_max - pd.Timedelta((i - 1) * n_days_test, 'd')
        train_index_start = X_val.index.get_slice_bound(date_start_train, side='left')
        train_index_end = X_val.index.get_slice_bound(date_end_train, side='right')
        test_index_start = X_val.index.get_slice_bound(date_start_test, side='left')
        test_index_end = X_val.index.get_slice_bound(date_end_test, side='right')
        train_index = indices[train_index_start:train_index_end]
        test_index = indices[test_index_start:test_index_end]
        indices_list.append((train_index, test_index))
        dates_list.append((date_start_train, date_end_train, date_start_test, date_end_test))
    
    if return_dates:
        return indices_list, dates_list
    else:
        return indices_list

# Optuna study
def opt_objective(trial, train_set, cv_iter, cv_dates, params, fobj, feval, df_Sc, df_St):
    # Define trial params and add default_params
    trial_params = {
        'feature_pre_filter':False,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 8, 1024),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 5000, log=True),
    }
    trial_params.update(params)
    if 'flag_params_random_hierarchical_loss' in params:
        trial_params_random_loss = {
            'max_levels_random': trial.suggest_int('max_levels_random', 2, 10),
            'max_categories_per_random_level': trial.suggest_int('max_categories_per_random_level', 2, 1000),
            'hier_freq': trial.suggest_int('hier_freq', 1, 10)
        }
        trial_params.update(trial_params_random_loss)
    # Perform cross-validation using walk-forward validation
    X, y = train_set.data, train_set.get_label()
    Sc = csc_matrix(df_Sc.sparse.to_coo())
    n_folds = len(cv_iter)
    best_iter, best_score = 0, 0.0
    for (train_index, val_index), (date_start_train, date_end_train, date_start_test, date_end_test) in zip(cv_iter, cv_dates):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]
        train_set_fold = lgb.Dataset(X_train, y_train)
        eval_set_fold = lgb.Dataset(X_val, y_val)
        df_St_train_fold = df_St.loc[:, date_start_train:date_end_train]
        df_St_val_fold = df_St.loc[:, date_start_test:date_end_test]
        St_train_fold = csc_matrix(df_St_train_fold.T.sparse.to_coo())
        St_val_fold = csc_matrix(df_St_val_fold.T.sparse.to_coo())
        trial_params, obj, eval = set_objective_metric(trial_params, fobj, feval, Sc, 
                                                 St_train=St_train_fold, St_val=St_val_fold)

        if trial_params['objective'] == 'tweedie':
            trial_params_tweedie = {'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.1, 1.9)}
            trial_params.update(trial_params_tweedie)

        model = lgb.train(params = trial_params,
                          num_boost_round = params['n_estimators'], 
                          train_set = train_set_fold,
                          valid_sets = [eval_set_fold],
                          fobj = obj,
                          feval = eval,
                          callbacks=[early_stopping(100), 
                                log_evaluation(100)],
                          )

        best_iter += (1 / n_folds) * model.best_iteration
        best_score += (1 / n_folds) * model.best_score['valid_0'][trial_params['metric']]

    trial.set_user_attr("best_iter", int(best_iter))

    return best_score

def get_best_params(params, param_filename, train_set, fobj, feval, df_Sc, df_St):
    param_file = Path(param_filename)
    if params['tuning'] and not param_file.is_file():
        # Create validation set
        time_index = train_set.data.index.name
        cv_iter, cv_dates = cv_iterator(train_set.data, time_index, params['n_validation_sets'], params['n_days_test'], return_dates=True)
        # Create Optuna study and run hyperparameter optimization
        sampler = optuna.samplers.TPESampler(seed=params['seed'])
        study = optuna.create_study(sampler=sampler, direction="minimize")
        wrapped_opt_opjective = lambda trial: opt_objective(trial, train_set, cv_iter, cv_dates, params, fobj, feval, df_Sc, df_St)
        study.optimize(wrapped_opt_opjective, n_trials=params['n_trials'])
        joblib.dump(study, param_filename)
    # Load best parameters if they exist, else use default values
    if param_file.is_file():
        print('Using optimized params')
        study = joblib.load(param_filename)
        params.update(study.best_params)
        params['n_estimators'] = study.best_trial.user_attrs['best_iter']
    else:
        print('Using default params')
    
    return params

#%% Setting 1: A single global model that forecasts all timeseries (bottom level AND aggregates)
def exp_m5_globalall(X, Xind, targets, target, time_index, end_train, df_Sc, df_St, 
                    exp_name, exp_folder, params, fobj=None, feval=None, seed=0):
    # Set parameters
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    if fobj == None or fobj == 'l2':
        params['objective'] = 'l2'
        fobj = None
    elif fobj == 'tweedie':
        params['objective'] = 'tweedie'
        fobj = None
    params['metric'] = 'l2'
    # Create train set
    y_train = X[target].loc[:end_train]
    X_train = X.drop(columns=[target]).loc[:end_train]
    train_set = lgb.Dataset(X_train, y_train)
    # Tune if required
    param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_best_params.params'
    params = get_best_params(params, param_filename, train_set, fobj, feval, df_Sc, df_St)
    # Train & save model
    model = lgb.train(params, train_set)
    joblib.dump(model, f'./src/exp_m5/{exp_folder}/{exp_name}_model.pkl')
    # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
    yhat = model.predict(X.drop(columns=[target]))
    yhat = np.clip(yhat, 0, 1e9).astype('float32')
    df_yhat = pd.Series(index=Xind, data=yhat)
    forecasts = df_yhat.unstack([time_index]).loc[targets.index, X.index.get_level_values(time_index).unique()]

    return forecasts
#%% Setting 2: A separate model for each aggregation in the hierarchy
def exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, df_Sc, df_St, 
                    exp_name, exp_folder, params, fobj=None, feval=None, seed=0):
    # Create parameter dict
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    if fobj == None or fobj == 'l2':
        params['objective'] = 'l2'
    elif fobj == 'tweedie':
        params['objective'] = 'tweedie'
    params['metric'] = 'l2'
    # Loop over aggregations
    forecasts_levels = []
    default_params = params.copy()
    for level in df_Sc.index.get_level_values('Aggregation').unique():
        print(f'Training level: {level}')
        params_level = default_params.copy()
        # Only keep bottom-level timeseries
        Xl_ind = pd.DataFrame(index=Xind).loc[level].index
        Xl = X[X['Aggregation'] == level]
        # Create train set
        y_train = Xl[target].loc[:end_train]
        X_train = Xl.drop(columns=[target]).loc[:end_train]
        train_set = lgb.Dataset(X_train, y_train)    
        # Tune if required
        param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_{level}_best_params.params'
        params_level = get_best_params(params_level, param_filename, train_set, fobj, feval, df_Sc, df_St)
        # Train & save model
        model = lgb.train(params_level, train_set)
        joblib.dump(model, f'./src/exp_m5/{exp_folder}/{exp_name}_{level}_model.pkl')
        # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
        yhat = model.predict(Xl.drop(columns=[target]))
        yhat = np.clip(yhat, 0, 1e9).astype('float32')
        df_yhat = pd.Series(index=Xl_ind, data=yhat)
        forecasts_level = df_yhat.unstack([time_index]).loc[targets.loc[level].index, Xl.index.get_level_values(time_index).unique()]
        forecasts_levels.append(pd.concat({f'{level}': forecasts_level}, names=['Aggregation']))

    forecasts = pd.concat(forecasts_levels)

    return forecasts
#%% Setting 3: A single global model that forecasts ONLY the bottom level timeseries
def exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, name_bottom_timeseries, 
                            df_Sc, df_St,  exp_name, exp_folder, params, fobj=None, feval=None, seed=0):
    # Only keep bottom-level timeseries
    Xb_ind = pd.DataFrame(index=Xind).loc[name_bottom_timeseries].index
    Xb = X[X['Aggregation'] == name_bottom_timeseries]
    # Convert df_S 
    S = csc_matrix(df_Sc.sparse.to_coo())
    # Create parameter dict
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    # Set objective
    if fobj == None or fobj == 'l2':
        params['objective'] = 'l2'
        fobj = None
    elif fobj == 'tweedie':
        params['objective'] = 'tweedie'
        fobj = None
    elif fobj == 'hierarchical_obj_se':
        params['objective'] = None
        fobj = partial(hierarchical_obj_se, S=S)
    elif fobj == 'hierarchical_obj_se_random':
        params['objective'] = None
        params['flag_params_random_hierarchical_loss'] = True
        fobj = partial(hierarchical_obj_se_random, S=S)
    # Set eval metric
    if feval is None or feval == 'l2':
        params['metric'] = 'l2'
        feval = None
    elif feval == 'hierarchical_eval_hmse':
        params['metric'] = feval
        feval = partial(hierarchical_eval_mse, S=S)
    elif feval == 'tweedie':
        params['metric'] = 'tweedie'
        feval = None
    # Create train set
    y_train = Xb[target].loc[:end_train]
    X_train = Xb.drop(columns=[target]).loc[:end_train]
    # Add attributes for hierarchical loss
    train_set = lgb.Dataset(X_train, y_train)
    params['n_bottom_timeseries'] = S.shape[1]
    # Tune if required
    param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_best_params.params'
    params = get_best_params(params, param_filename, train_set, fobj, feval, df_Sc, df_St)
    # Train & save model
    model = lgb.train(params, train_set, fobj=fobj)
    joblib.dump(model, f'./src/exp_m5/{exp_folder}/{exp_name}_model.pkl')
    # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
    yhat = model.predict(Xb.drop(columns=[target]))
    yhat = np.clip(yhat, 0, 1e9).astype('float32')
    df_yhat = pd.Series(index=Xb_ind, data=yhat)
    forecasts_bu_bottom_level = df_yhat.unstack([time_index]).loc[targets.loc[name_bottom_timeseries].index, Xb.index.get_level_values(time_index).unique()]
    # Aggregate bottom-up forecasts
    forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_Sc, name_bottom_timeseries)

    return forecasts_bu

#%% Setting 3a: A single global model that forecasts ONLY the bottom level timeseries, with temporal hierarchies added
def exp_m5_globalbottomup_withtemp(X, Xind, targets, target, time_index, end_train, name_bottom_timeseries, 
                            df_Sc, df_St, exp_name, exp_folder, params, fobj=None, feval=None, seed=0):
    # Only keep bottom-level timeseries
    Xb_ind = pd.DataFrame(index=Xind).loc[name_bottom_timeseries].index
    Xb = X[X['Aggregation'] == name_bottom_timeseries]
    # Convert df_S 
    Sc = csc_matrix(df_Sc.sparse.to_coo())
    St = csc_matrix(df_St.T.sparse.to_coo())
    # Create parameter dict
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    params['n_bottom_timeseries'] = Sc.shape[1]
    # Create train set
    y_train = Xb[target].loc[:end_train]
    X_train = Xb.drop(columns=[target]).loc[:end_train]
    # Add attributes for hierarchical loss
    train_set = lgb.Dataset(X_train, y_train)
    # Tune if required
    param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_best_params.params'
    params = get_best_params(params, param_filename, train_set, fobj, feval, df_Sc, df_St)
    # Train & save model
    params, obj, _ = set_objective_metric(params, fobj, feval, Sc, St_train=St)
    model = lgb.train(params, train_set, fobj=obj)
    joblib.dump(model, f'./src/exp_m5/{exp_folder}/{exp_name}_model.pkl')
    # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
    yhat = model.predict(Xb.drop(columns=[target]))
    yhat = np.clip(yhat, 0, 1e9).astype('float32')
    df_yhat = pd.Series(index=Xb_ind, data=yhat)
    forecasts_bu_bottom_level = df_yhat.unstack([time_index]).loc[targets.loc[name_bottom_timeseries].index, Xb.index.get_level_values(time_index).unique()]
    # Aggregate bottom-up forecasts
    forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_Sc, name_bottom_timeseries)

    return forecasts_bu

def set_objective_metric(params, fobj, feval, Sc, St_train, St_val=None):
    # Set objective
    if fobj == None or fobj == 'l2':
        params['objective'] = 'l2'
        obj = None
    elif fobj == 'tweedie':
        params['objective'] = 'tweedie'
        obj = None
    elif fobj == 'hierarchical_obj_se':
        params['objective'] = None
        obj = partial(hierarchical_obj_se, S=Sc)
    elif fobj == 'hierarchical_obj_se_withtemp':
        params['objective'] = None
        obj = partial(hierarchical_obj_se_withtemp, Sc=Sc, St=St_train)
    elif fobj == 'hierarchical_obj_se_random':
        params['objective'] = None
        params['flag_params_random_hierarchical_loss'] = True
        obj = partial(hierarchical_obj_se_random, S=Sc)
    # Set eval metric
    if feval is None or feval == 'l2':
        params['metric'] = 'l2'
        eval = None
    elif feval == 'hierarchical_eval_hmse':
        params['metric'] = feval
        eval = partial(hierarchical_eval_mse, S=Sc)
    elif feval == 'hierarchical_eval_hmse_withtemp':
        assert St_val is not None
        params['metric'] = feval
        eval = partial(hierarchical_eval_mse_withtemp, Sc=Sc, St=St_val)
    elif feval == 'tweedie':
        params['metric'] = 'tweedie'
        eval = None

    return params, obj, eval
