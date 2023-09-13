#%% Import packages
import time
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive, CrostonOptimized)
from hierts.reconciliation import apply_reconciliation_methods, hierarchy_temporal, hierarchy_cross_sectional
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
from helper_functions import read_m5, get_aggregations, create_forecast_set
#%% Set experiment parameters
if __name__ == "__main__":
    seed = 0
    # store_level = True
    store_level = False
    store_id = 0
    learning_rate = 0.05
    if store_level:
        exp_folder = f"exp1_storeid={store_id}/lr{learning_rate}"
    else:
        exp_folder = f"exp2_allstores/lr{learning_rate}"
    assert CURRENT_PATH.joinpath(exp_folder).is_dir()
    cross_sectional_aggregations, temporal_aggregations = get_aggregations(store_level)
    time_index = 'date'
    target = 'sales'
    name_bottom_timeseries = 'products'
    end_train = pd.to_datetime('2016-04-24')
    start_test = pd.to_datetime('2016-04-25')
    # Other experiment settings
    n_seeds = 10
    default_params = {'seed': 0,
                    'n_estimators': 2000,
                    'n_trials': 100,
                    'learning_rate': 0.05,
                    'verbosity': -1,
                    'tuning': True,
                    'n_validation_sets': 3,
                    'max_levels_random': 2,
                    'max_categories_per_random_level': 5,
                    'n_days_test': 28,
                    'n_years_train': 3,
                    'reset_feature_fraction': False,
                    'reset_feature_fraction_value': 1.0}
    #%% Read data
    df = read_m5(store_level=store_level, store_id=store_id)
    # Add columns for temporal hierarchies
    df['week'] = df[time_index].dt.isocalendar().week
    df['year'] = df[time_index].dt.year
    df['month'] = df[time_index].dt.month
    df['day'] = df[time_index].dt.day
    # Calculate cross-sectional and temporal hierarchy summing matrices
    df_Sc = hierarchy_cross_sectional(df, cross_sectional_aggregations, sparse=True, name_bottom_timeseries=name_bottom_timeseries)
    df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)
    # Create forecast set
    aggregation_cols = list(dict.fromkeys([col for cols in cross_sectional_aggregations for col in cols]))
    df = df.drop(columns = ['week', 'year', 'month', 'day'])
    X, Xind, targets = create_forecast_set(df, df_Sc, aggregation_cols, time_index, target)
    #%% Initiate experiment
    seasonality_length = 7 # Represents weekly data

    X['unique_id'] = X['Aggregation'].astype(str) + '_' + X['Value'].astype(str)
    dates = X.index.get_level_values(time_index).unique().sort_values()
    Y_df = X.reset_index().rename(columns = {time_index: 'ds', target: 'y'})
    Y_df = Y_df[["unique_id", "Aggregation", "Value", "ds", "y"]]
    df_train = Y_df[["unique_id", "ds", "y"]]

    models = [
        AutoARIMA(season_length=seasonality_length),
        AutoETS(season_length=seasonality_length),
        AutoTheta(season_length=seasonality_length),
        Naive(),
        SeasonalNaive(season_length=seasonality_length),
        CrostonOptimized(),
        ]

    # Loop over models
    for model in models:
        df_result = pd.DataFrame()
        df_result_timings = pd.DataFrame()
        sf = StatsForecast(
            df=df_train[(df_train["ds"] <= end_train)], 
            models=[model],
            freq='D', 
            n_jobs=-1,
            fallback_model = SeasonalNaive(season_length=seasonality_length),
        )
        model_name = str(sf.models[0])
        # Fit models
        start = time.perf_counter()      
        yhat_outsample = sf.forecast(h = 28, fitted=True)
        yhat_insample = sf.forecast_fitted_values()
        # Combine
        yhat = pd.concat((yhat_insample.drop(columns="y"), yhat_outsample), axis=0)
        yhat = yhat.merge(Y_df[["ds", "Aggregation", "Value", "unique_id"]], how="left", left_on=["ds", "unique_id"], right_on=["ds", "unique_id"])
        yhat = yhat.drop(columns = "unique_id")
        yhat = yhat.dropna()
        yhat[model_name] = np.clip(yhat[model_name], 0, 1e9).astype('float32')
        yhat = yhat.rename(columns = {"ds": time_index})
        start_date = yhat[time_index].min()
        yhat = yhat.set_index(["Aggregation", "Value", time_index])
        forecasts = yhat.unstack([time_index]).loc[targets.index]
        # Apply reconciliation methods
        forecasts.columns = forecasts.columns.droplevel(level=0)
        forecasts_test = forecasts.loc[:, start_test:]       
        forecasts_methods, t_reconciliation_seed = apply_reconciliation_methods(forecasts_test, df_Sc, targets.loc[:, start_date:end_train], forecasts.loc[:, start_date:end_train],
                        methods = ['ols', 'wls_struct', 'wls_var', 'mint_shrink', 'erm'], positive=True, return_timing=True)
        end = time.perf_counter()
        t_trainpredict_seed = (end - start)
        # Add result to result df
        dfc = pd.concat({f'{seed}': forecasts_methods}, names=['Seed'])
        df_result = pd.concat((df_result, dfc))
        # Add timings to timings df
        df_seed = pd.DataFrame({'t_trainpredict':t_trainpredict_seed}, index=[seed])
        df_reconciliation = pd.DataFrame(t_reconciliation_seed, index=[seed])
        df_result_timings = pd.concat((df_result_timings, pd.concat((df_seed, df_reconciliation), axis=1)))
        # Save
        df_result.columns = df_result.columns.astype(str)
        df_result.to_parquet(str(CURRENT_PATH.joinpath(f"{exp_folder}/{model_name}_.parquet")))
        df_result_timings.to_csv(str(CURRENT_PATH.joinpath(f"{exp_folder}/{model_name}_timings.csv")))