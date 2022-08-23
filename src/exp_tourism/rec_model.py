#%% Read packages
import pandas as pd
import numpy as np
from src.lib import calc_summing_matrix, apply_reconciliation_methods, aggregate_bottom_up_forecasts, calc_level_method_rmse
from sktime.forecasting.ets import AutoETS
#%% Read data
df = pd.read_csv('src/exp_tourism/tourism.csv', index_col=0)
df['Quarter'] = pd.PeriodIndex(df['Quarter'].str[0:4] + '-' + df['Quarter'].str[5:], freq='q')
#%% Set aggregations and target
aggregation_cols = ['State', 'Region', 'Purpose']
aggregations = [['State'],
                ['State', 'Region'],
                ['State', 'Purpose'],
                ['Purpose']]
target = 'Trips'
time_index = 'Quarter'
end_train = '2015Q4'
start_test = '2016Q1'
#%% Calculate summing matrix
df_S = calc_summing_matrix(df, aggregation_cols, aggregations)
#%% Create a forecasting model for each time series in the aggregation matrix df_S
df['bottom_timeseries'] = df[aggregation_cols].agg('-'.join, axis=1)
dfc = df.set_index(['bottom_timeseries', time_index])[target].unstack(0)
dfc = dfc[df_S.columns]
forecasts = pd.DataFrame(index=df_S.index, columns = dfc.index, dtype=np.float32)
actuals = pd.DataFrame(index=df_S.index, columns = dfc.index, dtype=np.float32)
for aggregate, summing_vector in df_S.iterrows():
    # Get series
    series = dfc @ summing_vector
    # Fit model and predict (we need to clip because otherwise there's a convergence error)
    model = AutoETS(auto=True, n_jobs=1, random_state=0)
    model.fit(np.clip(series.loc[:end_train], 1e-3, 1e16))
    forecast = model.predict(series.index)
    # Store to forecasts/actuals array
    forecasts.loc[aggregate] = forecast.values
    actuals.loc[aggregate] = series.values

residuals = (forecasts - actuals)
#%% Reconciliation
# All forecasts
residuals_train = residuals.loc[:, :end_train]
forecasts_test = forecasts.loc[:, start_test:]
forecasts_methods = apply_reconciliation_methods(forecasts_test, df_S, residuals_train, methods=['ols', 'wls_struct', 'wls_var', 'mint_shrink'])
# Bottom-up forecasts
forecasts_bu_bottom_level = forecasts.loc['Bottom level']
forecasts_bu = aggregate_bottom_up_forecasts(forecasts_bu_bottom_level, df_S)
residuals_bu = (forecasts_bu - actuals)
forecasts_bu_test = forecasts_bu.loc[:, start_test:]
forecasts_method = pd.concat({'bottom-up': forecasts_bu_test}, names=['Method'])
forecasts_methods = pd.concat((forecasts_method, forecasts_methods), axis=0)
# Calculate error for all levels and methods
rmse, rel_rmse = calc_level_method_rmse(forecasts_methods, actuals, base='bottom-up')