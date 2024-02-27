import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import joblib
import time
import warnings
import sys
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
sys.path.append(str(CURRENT_PATH.parents[1]))
from hierts.reconciliation import aggregate_bottom_up_forecasts
from scipy.sparse import csc_matrix, vstack, eye
from lightgbm import early_stopping, log_evaluation
from functools import partial
from src.lib import prepare_HierarchicalLoss, prepare_RandomHierarchicalLoss, HierarchicalLossObjective, HierarchicalLossMetric, RandomHierarchicalLossObjective
warnings.filterwarnings('ignore')
#%% Setting 1: A single global model that forecasts all timeseries (bottom level AND aggregates)
def exp_m5_globalall(X, Xind, targets, target, time_index, end_train, start_test, df_Sc, df_St, 
                    exp_name, exp_folder, params, sobj=None, seval=None, seed=0):
    # Set parameters
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    # Create train set for cross-validation
    y_train = X[target].loc[:end_train]
    X_train = X.drop(columns=[target]).loc[:end_train]
    # Tune if required
    param_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_best_params.params")
    params = get_best_params(params, param_filepath, X_train, y_train, exp_name, sobj, seval, df_Sc, df_St)
    # Create training set for final model
    start_train = pd.Timestamp(end_train) - pd.Timedelta(params['n_years_train'] * 366, 'd')
    y_train = X[target].loc[start_train:end_train]
    X_train = X.drop(columns=[target]).loc[start_train:end_train]
    train_set = lgb.Dataset(X_train, y_train)
    # Set objective and metric functions
    params, fobj = set_objective(params, exp_name, sobj)    
    # Train and save final model
    model_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_model_seed_{seed}.pkl")
    if not model_filepath.is_file():
        print("No pre-trained model found. Training model...")
        start = time.perf_counter()
        model = lgb.train(params, train_set, fobj=fobj)
        end = time.perf_counter()
        t_train = (end - start)
        joblib.dump(model, str(model_filepath))
    else:
        print("Loading pre-trained model")
        model = joblib.load(str(model_filepath))
        timings_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_timings.csv")
        df_timings = pd.read_csv(str(timings_filepath), index_col=0)
        t_train = df_timings.loc[seed]["t_train"]
    # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
    start = time.perf_counter()
    dates = X.index.get_level_values(time_index).unique().sort_values()
    df_yhat = get_predictions(model, X, target, time_index, end_train, start_test)
    forecasts = df_yhat.unstack([time_index]).loc[targets.index, dates]
    end = time.perf_counter()
    t_predict = (end - start)

    return forecasts, t_train, t_predict

#%% Setting 2: A separate model for each aggregation in the hierarchy
def exp_m5_sepagg(X, Xind, targets, target, time_index, end_train, start_test, df_Sc, df_St, 
                    exp_name, exp_folder, params, sobj=None, seval=None, seed=0):
    # Create parameter dict
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    # Loop over aggregations
    forecasts_levels = []
    default_params = params.copy()
    t_train, t_predict = 0.0, 0.0
    for level in df_Sc.index.get_level_values('Aggregation').unique():
        print(f'Training level: {level}')
        params_level = default_params.copy()
        # Only keep bottom-level timeseries
        Xl_ind = pd.DataFrame(index=Xind).loc[level].index
        Xl = X[X['Aggregation'] == level]
        # Create train set for cross-validation
        y_train = Xl[target].loc[:end_train]
        X_train = Xl.drop(columns=[target]).loc[:end_train]
        # Tune if required
        param_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_{level}_best_params.params")
        params_level = get_best_params(params_level, param_filepath, X_train, y_train, exp_name, sobj, seval, df_Sc, df_St)
        # Create training set for final model
        start_train = pd.Timestamp(end_train) - pd.Timedelta(params['n_years_train'] * 366, 'd')
        y_train = Xl[target].loc[start_train:end_train]
        X_train = Xl.drop(columns=[target]).loc[start_train:end_train]
        train_set = lgb.Dataset(X_train, y_train)    
        # Set objective and metric functions
        params_level, fobj = set_objective(params_level, exp_name, sobj)    
        # Train and save final model
        model_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_{level}_model_seed_{seed}.pkl")
        if not model_filepath.is_file():
            print("No pre-trained model found. Training model...")
            start = time.perf_counter()
            model = lgb.train(params_level, train_set, fobj=fobj)
            end = time.perf_counter()
            t_train += (end - start)
            joblib.dump(model, str(model_filepath))
        else:
            print("Loading pre-trained model")
            model = joblib.load(str(model_filepath))
            timings_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_timings.csv")
            df_timings = pd.read_csv(str(timings_filepath), index_col=0)
            t_train += df_timings.loc[seed]["t_train"]            
        # Make predictions for both train and test set (we need the train residuals for covariance estimation in the reconciliation methods)
        start = time.perf_counter()
        dates = Xl.index.get_level_values(time_index).unique().sort_values()
        df_yhat_level = get_predictions(model, Xl, target, time_index, end_train, start_test)
        df_yhat_level = df_yhat_level.droplevel(level="Aggregation", axis=0)
        forecasts_level = df_yhat_level.unstack([time_index]).loc[targets.loc[level].index, dates]
        forecasts_levels.append(pd.concat({f'{level}': forecasts_level}, names=['Aggregation']))
        end = time.perf_counter()
        t_predict += (end - start)

    forecasts = pd.concat(forecasts_levels)

    return forecasts, t_train, t_predict

#%% Setting 3: A single global model that forecasts ONLY the bottom level timeseries, with temporal hierarchies added
def exp_m5_globalbottomup(X, Xind, targets, target, time_index, end_train, start_test, name_bottom_timeseries, 
                            df_Sc, df_St, exp_name, exp_folder, params, sobj=None, seval=None, seed=0):
    # Only keep bottom-level timeseries
    Xb = X[X['Aggregation'] == name_bottom_timeseries]
    # Create parameter dict
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    params['n_bottom_timeseries'] = df_Sc.shape[1]
    # Create train set for cross-validation
    y_train = Xb[target].loc[:end_train]
    X_train = Xb.drop(columns=[target]).loc[:end_train]
    # Tune if required
    param_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_best_params.params")
    params = get_best_params(params, param_filepath, X_train, y_train, exp_name, sobj, seval, df_Sc, df_St)
    # Create training set for final model
    start_train = pd.Timestamp(end_train) - pd.Timedelta(params['n_years_train'] * 366, 'd')
    y_train = Xb[target].loc[start_train:end_train]
    X_train = Xb.drop(columns=[target]).loc[start_train:end_train]
    train_set = lgb.Dataset(X_train, y_train)
    # Create St and Sc
    df_St_train = df_St.loc[:, start_train:end_train]
    df_Sc_train = df_Sc
    # Set objective and metric functions
    params, fobj = set_objective(params, exp_name, sobj, df_Sc_train, df_St_train, seed=seed)
    # Train and save model
    model_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_model_seed_{seed}.pkl")
    if not model_filepath.is_file():
        print("No pre-trained model found. Training model...")
        if params['reset_feature_fraction'] == True:
            params['feature_fraction'] = params['reset_feature_fraction_value']
        start = time.perf_counter()
        model = lgb.train(params, train_set, fobj=fobj)
        end = time.perf_counter()
        t_train = (end - start)    
        joblib.dump(model, str(model_filepath))
    else:
        print("Loading pre-trained model")
        model = joblib.load(str(model_filepath))
        timings_filepath = CURRENT_PATH.joinpath(f"{exp_folder}/{exp_name}_timings.csv")
        df_timings = pd.read_csv(str(timings_filepath), index_col=0)
        t_train = df_timings.loc[seed]["t_train"]
    # Make predictions for only train set
    start = time.perf_counter()
    dates = Xb.loc[start_test:].index.get_level_values(time_index).unique().sort_values()
    df_yhat = get_predictions(model, Xb, target, time_index, end_train, start_test, insample=False)
    df_yhat = df_yhat.droplevel(level="Aggregation", axis=0)
    forecasts_bu_bottom_level = df_yhat.unstack([time_index]).loc[targets.loc[name_bottom_timeseries].index, dates]
    forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_Sc, name_bottom_timeseries)
    end = time.perf_counter()
    t_predict = (end - start)    

    return forecasts_bu, t_train, t_predict

#%% Objective and metric helper functions
def set_objective(params, exp_name, sobj, df_Sc=None, df_St=None, seed=0):
    # Set objective
    if sobj == None or sobj == 'l2':
        params['objective'] = 'l2'
        fobj = None
    elif sobj == 'tweedie':
        params['objective'] = 'tweedie'
        fobj = None
    elif sobj == 'hierarchical_obj_se':
        assert df_Sc is not None
        assert df_St is not None
        params['objective'] = None
        n_bottom_timeseries = df_Sc.shape[1]
        n_bottom_timesteps = df_St.shape[1]
        if exp_name == 'bu_objhse_evalhmse' or exp_name == 'bu_objhse_evalmse':
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=df_Sc,
                                                         df_St=None)
            St = None
        elif exp_name == 'bu_objhse_evalhmse_withtemponly':
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=None,
                                                         df_St=df_St)
            Sc = None
        elif exp_name == 'bu_objhse_evalhmse_withtemp':
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=df_Sc,
                                                         df_St=df_St)
        elif exp_name == 'bu_objhse_evalhmse_random':
            max_levels_random = params['max_levels_random']
            max_categories_per_random_level = params['max_categories_per_random_level']
            rng = np.random.default_rng(seed=seed)
            hessian, denominator, Sc, Scd, St, Std = prepare_RandomHierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         max_levels_random=max_levels_random,
                                                         max_categories_per_random_level=max_categories_per_random_level,
                                                         rng=rng)
            St = None
        elif exp_name == 'bu_objhse_evalhmse_softmax':
            softmax_temperature = params['softmax_temperature']
            Sc = df_Sc.sparse.to_coo().tocsr().astype(np.float32)



            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=df_Sc,
                                                         df_St=None)
            St = None            
            rng = np.random.default_rng(seed=seed)
            permutation_rate = params['permutation_rate']
            Sc = df_Sc.sparse.to_coo().tocsr().astype(np.float32)
            Sc_levels = Sc[1:-n_bottom_timeseries]
            ix = rng.choice([True, False], size=Sc_levels.shape, replace=True, p=[permutation_rate, 1 - permutation_rate])
            Sc

        else:
            raise NotImplementedError

        fobj = partial(HierarchicalLossObjective, hessian=hessian, 
                    n_bottom_timeseries=n_bottom_timeseries, n_bottom_timesteps=n_bottom_timesteps, 
                        Sc=Sc, St=St, Scd=Scd, Std=Std)
    elif sobj == 'hierarchical_obj_se_random':
        assert df_Sc is not None
        params['objective'] = None
        n_bottom_timeseries = df_Sc.shape[1]
        max_levels_random = params['max_levels_random']
        max_categories_per_random_level = params['max_categories_per_random_level']
        hier_freq = params['hier_freq']       
        rng = np.random.default_rng(seed=seed)
        fobj = partial(RandomHierarchicalLossObjective, rng=rng,
                       n_bottom_timeseries=n_bottom_timeseries, max_levels_random=max_levels_random, 
                       max_categories_per_random_level=max_categories_per_random_level, 
                       hier_freq=hier_freq)
                
    return params, fobj

def set_metric(params, exp_name, seval, df_Sc=None, df_St=None):
    # Set eval metric
    if seval is None or seval == 'l2':
        params['metric'] = 'l2'
        feval = None
    elif seval == 'hierarchical_eval_hmse':
        assert df_Sc is not None
        assert df_St is not None
        params['metric'] = 'hierarchical_eval_hmse'
        n_bottom_timeseries = df_Sc.shape[1]
        n_bottom_timesteps = df_St.shape[1]

        if (exp_name == 'bu_objhse_evalhmse' 
            or exp_name == 'bu_objrhse_evalhmse' 
            or exp_name == 'bu_objmse_evalhmse' 
            or exp_name == 'bu_objtweedie_evalhmse'):
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=df_Sc,
                                                         df_St=None)
            St = None
        elif exp_name == 'bu_objhse_evalhmse_withtemponly':
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=None,
                                                         df_St=df_St)
            Sc = None
        elif exp_name == 'bu_objhse_evalhmse_withtemp':
            hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(n_bottom_timeseries=n_bottom_timeseries, 
                                                         n_bottom_timesteps=n_bottom_timesteps, 
                                                         df_Sc=df_Sc,
                                                         df_St=df_St)
        else:
            raise NotImplementedError
        
        feval = partial(HierarchicalLossMetric, denominator=denominator,  
                       n_bottom_timeseries=n_bottom_timeseries, n_bottom_timesteps=n_bottom_timesteps, 
                       Sc=Sc, St=St)       
     
    elif seval == 'tweedie':
        params['metric'] = 'tweedie'
        feval = None

    return params, feval
#%% Hyperparameter tuning helper functions
def get_best_params(params, param_filepath, X, y, exp_name, sobj, seval, df_Sc, df_St):
    if params['tuning'] and not param_filepath.is_file():
        # Create validation set
        time_index = X.index.name
        cv_iter = cv_iterator(X, 
                                time_index, 
                                params['n_validation_sets'], 
                                params['n_days_test'], 
                                params['n_years_train'] )
        # Create Optuna study and run hyperparameter optimization
        sampler = optuna.samplers.TPESampler(seed=params['seed'])
        study = optuna.create_study(sampler=sampler, direction="minimize")
        wrapped_opt_opjective = lambda trial: opt_objective(trial, X, y, cv_iter, params, exp_name, sobj, seval, df_Sc, df_St)
        study.optimize(wrapped_opt_opjective, n_trials=params['n_trials'])
        joblib.dump(study, str(param_filepath))
    # Load best parameters if they exist, else use default values
    if param_filepath.is_file():
        print('Using optimized params')
        study = joblib.load(str(param_filepath))
        params.update(study.best_params)
        params['n_estimators'] = study.best_trial.user_attrs['best_iter']
    else:
        print('Using default params')
    
    return params

# Cross-validation iterator
def cv_iterator(X, time_index, n_splits=6, n_days_test=28, n_years_train=3):
    indices = np.arange(X.shape[0])
    date_max = X.index.get_level_values(time_index).max()
    indices_list = []
    for i in range(n_splits, 0, -1):
        date_end_train = date_max - pd.Timedelta(i * n_days_test, 'd')
        date_start_train = date_end_train - pd.Timedelta(n_years_train * 366, 'd')
        date_start_test = date_end_train + pd.Timedelta(1, 'd')
        date_end_test = date_max - pd.Timedelta((i - 1) * n_days_test, 'd')
        train_index_start = X.index.get_slice_bound(date_start_train, side='left')
        train_index_end = X.index.get_slice_bound(date_end_train, side='right')
        test_index_start = X.index.get_slice_bound(date_start_test, side='left')
        test_index_end = X.index.get_slice_bound(date_end_test, side='right')
        train_index = indices[train_index_start:train_index_end]
        test_index = indices[test_index_start:test_index_end]
        indices_list.append((train_index, test_index))

    return indices_list

# Optuna study
def opt_objective(trial, X, y, cv_iter, params, exp_name, sobj, seval, df_Sc, df_St):
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
    # Set additional params for random hierarchical loss
    if sobj == 'hierarchical_obj_se_random':
        trial_params_random_loss = {
            'max_levels_random': trial.suggest_int('max_levels_random', 2, 10),
            'max_categories_per_random_level': trial.suggest_int('max_categories_per_random_level', 2, 1000),
            'hier_freq': trial.suggest_int('hier_freq', 1, 7)
        }
        trial_params.update(trial_params_random_loss)      
    # Set additional params for tweedie loss
    if sobj == 'tweedie':
        trial_params_tweedie = {'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.1, 1.9)}
        trial_params.update(trial_params_tweedie)
    best_score, best_iter = 0.0, 0
    n_folds = len(cv_iter)
    for train_index, val_index in cv_iter:
        # Create train and validation sets
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_val, y_val)
        # Create St and Sc
        df_St_train = df_St.loc[:, X_train.index.min():X_train.index.max()]
        df_St_val = df_St.loc[:, X_val.index.min():X_val.index.max()]
        df_Sc_train = df_Sc
        df_Sc_val = df_Sc
        # Set objective and metric functions
        trial_params, fobj = set_objective(trial_params, exp_name, sobj, df_Sc_train, df_St_train)
        trial_params, feval = set_metric(trial_params, exp_name, seval, df_Sc_val, df_St_val)
        # Train model
        model = lgb.train(params = trial_params,
                          train_set = train_set, 
                          valid_sets = [valid_set],
                          num_boost_round = trial_params['n_estimators'],
                          fobj=fobj,
                          feval=feval,
                          callbacks=[early_stopping(100), 
                                     log_evaluation(100)],
                          verbose_eval=False
                          )
        # Save best iteration and score
        best_iter += (1 / n_folds) * model.best_iteration
        best_score += (1 / n_folds) * model.best_score['valid_0'][trial_params['metric']]

    # Return best score, add best iteration to trial attributes
    trial.set_user_attr("best_iter", int(best_iter))

    return best_score

def get_predictions(model, X, target, time_index, end_train, start_test, insample=True):
    # in-sample predictions
    if insample:
        X_test_current = X.drop(columns=[target]).loc[:end_train]
        yhat_insample = model.predict(X_test_current)
        yhat_insample = np.clip(yhat_insample, 0, 1e9).astype('float32')
        Xind_current = X_test_current[["Aggregation", "Value"]].reset_index().set_index(["Aggregation", "Value", time_index]).index
        df_yhat_insample = pd.Series(index = Xind_current, data=yhat_insample)
    # out-of-sample predictions
    n_dates_test = X.loc[start_test:].index.nunique()
    X_test = X.drop(columns = [target])
    cols = X_test.columns.copy()
    lag_columns = [col for col in X_test.columns if 'lag' in col and not 'mavg' in col]
    lag_columns = [col for col in lag_columns if int(col.rsplit('lag', 1)[1]) < n_dates_test]
    mavg_columns = [col for col in X_test.columns if 'mavg' in col]
    mavgs = [col.rsplit('mavg', 1)[1] for col in mavg_columns]
    other_columns = [col for col in X_test.columns if 
                       col not in lag_columns 
                       and col not in mavg_columns 
                       and 'Aggregation' not in col 
                       and 'Value' not in col]
    # Pandas is crap software, some idiot thought it's a good idea to create a transposed Series from a DataFrame if you only select a single row
    X_test_lags = X_test.loc[[start_test], lag_columns]
    X_test_mavgs = X_test.loc[[start_test], mavg_columns]

    for date in range(n_dates_test):
        # Create current X_test
        X_test_agg = X_test.loc[[start_test], ["Aggregation", "Value"]]
        X_test_lags = pd.DataFrame(X_test_lags.values, index=X_test_agg.index, columns=X_test_lags.columns)
        X_test_mavgs = pd.DataFrame(X_test_mavgs.values, index=X_test_agg.index, columns=X_test_mavgs.columns)
        X_test_other = X_test.loc[[start_test], other_columns]
        X_test_current = pd.concat((X_test_agg, X_test_lags, X_test_mavgs, X_test_other), axis=1)
        X_test_current = X_test_current[cols]
        # Predict
        yhat_outsample = model.predict(X_test_current)
        yhat_outsample = np.clip(yhat_outsample, 0, 1e9).astype('float32')
        Xind_current = X_test_current[["Aggregation", "Value"]].reset_index().set_index(["Aggregation", "Value", time_index]).index
        df_yhat_outsample_current = pd.Series(index = Xind_current, data=yhat_outsample)
        if date == 0:
            df_yhat_outsample = df_yhat_outsample_current
        else:
            df_yhat_outsample = pd.concat((df_yhat_outsample, df_yhat_outsample_current))
        # Fill lag columns with last prediction
        X_test_lags.loc[:, lag_columns[1:]] = X_test_lags.loc[:, lag_columns[:-1]].values
        X_test_lags.loc[:, lag_columns[0]] = yhat_outsample
        # Add last prediction to moving average
        for mavg, mavg_column in zip(mavgs, mavg_columns):
            X_test_mavgs.loc[:, mavg_column] -= (1 / int(mavg)) * X_test_mavgs.loc[:, mavg_column]
            X_test_mavgs.loc[:, mavg_column] += (1 / int(mavg)) * yhat_outsample
        # Go to next test point
        start_test += pd.DateOffset(days=1)

    # Output is combination of in- and out-of-sample predictions
    if insample:
        df_yhat = pd.concat((df_yhat_insample, df_yhat_outsample))
    else:
        df_yhat = df_yhat_outsample

    return df_yhat