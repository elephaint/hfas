"""
   Copyright (c) 2022- Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://github.com/elephaint/ralf/blob/main/LICENSE

"""
#%% Import packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
CURRENT_PATH = Path(__file__).parent
#%% Reduce memory
def reduce_mem(df):
    cols = df.columns
    for col in cols:
        col_dtype = df[col].dtype 
        try:               
            if col_dtype == pd.Int8Dtype():
                df[col] = df[col].astype('int8')
            elif col_dtype == pd.Int16Dtype():
                df[col] = df[col].astype('int16')
            elif col_dtype == pd.Int32Dtype():
                df[col] = df[col].astype('int16')    
            elif col_dtype == pd.Int64Dtype():
                df[col] = df[col].astype('int16')
            elif col_dtype == pd.Float64Dtype():
                df[col] = df[col].astype('float32')
        except:
            pass
        if col_dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif col_dtype == 'float64':
            if 'sales_lag' in col:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('float32')
            
    return df
#%% 1) Read datasets
df_calendar = pd.read_parquet('src/exp_m5/data/calendar.parquet')
df_sales = pd.read_parquet('src/exp_m5/data/sales_train_evaluation.parquet')
df_prices = pd.read_parquet('src/exp_m5/data/sell_prices.parquet')
#%% 2) Label encoding of categorical information sales data
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
df_itemids = df_sales[['id','item_id','dept_id','cat_id','store_id','state_id']].copy()
for col in df_itemids.columns:
    df_itemids[col+'_enc'] = LabelEncoder().fit_transform(df_itemids[col])
    df_itemids[col+'_enc'] = df_itemids[col+'_enc'].astype('int16')
    
df_itemids = df_itemids[['id_enc', 'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'store_id_enc', 'state_id_enc', 'id',
       'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
#%% 2a) Add label encoded item_ids to prices
df_prices = df_prices.merge(df_itemids[['store_id','item_id','item_id_enc','store_id_enc']], how='left', left_on=['store_id','item_id'], right_on=['store_id','item_id'])
# Assert that we don't create NaNs - in other words, every item in df_prices is represented in itemids
assert df_prices.isnull().sum().sum() == 0
df_prices = df_prices[['wm_yr_wk', 'sell_price', 'item_id_enc', 'store_id_enc']]
# Add weeks on sale
df_prices['weeks_on_sale'] = df_prices.groupby(['item_id_enc','store_id_enc']).cumcount() + 1
# Reduce mem
df_prices = reduce_mem(df_prices)
#%% 3) Label encoding of categorical information in calendar data
df_events = df_calendar[['event_name_1','event_type_1','event_name_2','event_type_2']].copy()
cols = df_events.columns
for col in cols:
    df_events[col+'_enc'] = LabelEncoder().fit_transform(df_events[col])
    df_events[col+'_enc'] = df_events[col+'_enc'].astype('int16')
df_events = df_events.drop(columns = cols)

df_events = pd.concat( (df_calendar[['date','d','wm_yr_wk','snap_CA', 'snap_TX','snap_WI']], df_events), axis = 1)
cols = df_events.columns[3:].tolist()
df_events[cols] = df_events[cols].astype('int8')
df_events['date'] = pd.to_datetime(df_events['date'])
# Reduce mem
df_events = reduce_mem(df_events)
#%% 4) Create main df
# Add zero columns for test period to sales data
new_columns = df_events['d'][-28:].tolist()
df_sales = df_sales.assign(**pd.DataFrame(data = 0, columns=new_columns, index=df_sales.index))
# Add item ids and stack
df = pd.concat((df_itemids['id_enc'], df_sales.iloc[:, 6:]), axis=1).convert_dtypes()
df = df.set_index(['id_enc'])
df = df.stack().reset_index()
df.columns = ['id_enc','d','sales']
# Reduce mem
df = reduce_mem(df)
#%% 5a) Add event and item features to main df
# Add event information
df = df.merge(df_events, how='left', left_on=['d'], right_on=['d'])
# Add item information
df = df.merge(df_itemids[['id','id_enc', 'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'store_id_enc', 'state_id_enc']], how='left', left_on=['id_enc'], right_on=['id_enc'])
# Reduce mem
df = reduce_mem(df)
#%% 5b) Add selling prices, fill nans.
df = df.merge(df_prices, how='left', right_on=['item_id_enc','store_id_enc','wm_yr_wk'], left_on = ['item_id_enc','store_id_enc','wm_yr_wk'])
df['sell_price'] = df['sell_price'].bfill()
df['weeks_on_sale'] = df['weeks_on_sale'].fillna(0)
#%% Change order of features - this is convenient for later on
# Everything up to 'date' is fixed for each day, thereafter is variable
cols =['sales',
 'date',
 'id',
 'id_enc',
 'item_id_enc',
 'dept_id_enc',
 'cat_id_enc',
 'store_id_enc',
 'state_id_enc',
 'snap_CA',
 'snap_TX',
 'snap_WI',
 'event_type_1_enc',
 'event_type_2_enc',
 'sell_price',
 'weeks_on_sale']

df = df[cols]
#%% 6) Create hdf with the stuff we need to keep
df['id'] = df['id'].astype('O')
df = df.sort_values(by=['store_id_enc','item_id_enc','date'])
df = df.reset_index(drop=True)
filename = 'm5_dataset_products.parquet'
filepath = CURRENT_PATH.joinpath(filename)
df.to_parquet(filepath)