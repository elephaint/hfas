import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from src.lib import hierarchical_obj_se, hierarchical_eval_mse, hierarchical_obj_se_random
from hierts.reconciliation import aggregate_bottom_up_forecasts
from scipy.sparse import csc_matrix
from lightgbm import early_stopping, log_evaluation
from functools import partial
#%% Hyperparameter tuning helper functions
# Cross-validation iterator
def cv_iterator(X_val, time_index, n_splits=6, n_days_test=28):
    indices = np.arange(X_val.shape[0])
    date_max = X_val.index.get_level_values(time_index).max()
    indices_list = []
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
    
    return indices_list

# Optuna study
def opt_objective(trial, train_set, cv_iter, params, fobj, feval):
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
    if params['objective'] == 'tweedie':
        trial_params_tweedie = {'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.1, 1.9)}
        trial_params.update(trial_params_tweedie)
    # Perform cross-validation using walk-forward validation
    cv_results = lgb.cv(trial_params,
                        train_set, 
                        num_boost_round=params['n_estimators'],
                        folds=cv_iter,
                        callbacks=[early_stopping(100), 
                                log_evaluation(100)],
                        fobj=fobj,
                        feval=feval)  
    # Return best score, add best iteration to trial attributes
    scores = cv_results[f"{trial_params['metric']}-mean"]
    best_score = scores[-1]
    trial.set_user_attr("best_iter", len(scores))

    return best_score

def get_best_params(params, param_filename, train_set, fobj, feval):
    param_file = Path(param_filename)
    if params['tuning'] and not param_file.is_file():
        # Create validation set
        time_index = train_set.data.index.name
        cv_iter = cv_iterator(train_set.data, time_index, params['n_validation_sets'], params['n_days_test'])
        # Create Optuna study and run hyperparameter optimization
        sampler = optuna.samplers.TPESampler(seed=params['seed'])
        study = optuna.create_study(sampler=sampler, direction="minimize")
        wrapped_opt_opjective = lambda trial: opt_objective(trial, train_set, cv_iter, params, fobj, feval)
        study.optimize(wrapped_opt_opjective, n_trials=params['n_trials'])
        joblib.dump(study, param_filename)
    # Load best parameters if they exist, else use default values
    if param_file.is_file():
        print('Using optimized params')
        study = joblib.load(param_filename)
        params.update(study.best_params)
        # params['n_estimators'] = study.best_trial.user_attrs['best_iter']
        params['n_estimators'] = 1641
    else:
        print('Using default params')
    
    return params

#%% Setting 1: A single global model that forecasts all timeseries (bottom level AND aggregates)
def exp_m5_globalall(X, Xind, targets, target, time_index, end_train, 
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
    params = get_best_params(params, param_filename, train_set, fobj, feval)
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
def exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, df_S, 
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
    for level in df_S.index.get_level_values('Aggregation').unique():
        print(f'Training level: {level}')
        # Only keep bottom-level timeseries
        Xl_ind = pd.DataFrame(index=Xind).loc[level].index
        Xl = X[X['Aggregation'] == level]
        # Create train set
        y_train = Xl[target].loc[:end_train]
        X_train = Xl.drop(columns=[target]).loc[:end_train]
        train_set = lgb.Dataset(X_train, y_train)    
        # Tune if required
        param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_{level}_best_params.params'
        params = get_best_params(params, param_filename, train_set, fobj, feval)
        # Train & save model
        model = lgb.train(params, train_set)
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
                            df_S, exp_name, exp_folder, params, fobj=None, feval=None, seed=0):
    # Only keep bottom-level timeseries
    Xb_ind = pd.DataFrame(index=Xind).loc[name_bottom_timeseries].index
    Xb = X[X['Aggregation'] == name_bottom_timeseries]
    # Convert df_S 
    S = csc_matrix(df_S.sparse.to_coo())
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
    params['n_levels'] = Xind.get_level_values('Aggregation').nunique()
    params['n_bottom_timeseries'] = S.shape[1]
    # Tune if required
    param_filename = f'./src/exp_m5/{exp_folder}/{exp_name}_best_params.params'
    params = get_best_params(params, param_filename, train_set, fobj, feval)
    # Train & save model
    model = lgb.train(params, train_set, fobj=fobj)
    joblib.dump(model, f'./src/exp_m5/{exp_folder}/{exp_name}_model.pkl')
    # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
    yhat = model.predict(Xb.drop(columns=[target]))
    yhat = np.clip(yhat, 0, 1e9).astype('float32')
    df_yhat = pd.Series(index=Xb_ind, data=yhat)
    forecasts_bu_bottom_level = df_yhat.unstack([time_index]).loc[targets.loc[name_bottom_timeseries].index, Xb.index.get_level_values(time_index).unique()]
    # Aggregate bottom-up forecasts
    forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_S, name_bottom_timeseries)

    return forecasts_bu
