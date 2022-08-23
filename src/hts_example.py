#%% hts example
from datetime import datetime
from hts import HTSRegressor
from hts.utilities.load_data import load_hierarchical_sine_data

# load some data
s, e = datetime(2020, 1, 1), datetime(2023, 1, 1)
hsd = load_hierarchical_sine_data(s, e).resample('1H').apply(sum)
hier = {'total': ['a', 'b', 'c'],
        'a': ['a_x', 'a_y'],
        'b': ['b_x', 'b_y'],
        'c': ['c_x', 'c_y'],
        'a_x': ['a_x_1', 'a_x_2'],
        'a_y': ['a_y_1', 'a_y_2'],
        'b_x': ['b_x_1', 'b_x_2'],
        'b_y': ['b_y_1', 'b_y_2'],
        'c_x': ['c_x_1', 'c_x_2'],
        'c_y': ['c_y_1', 'c_y_2']
    }

# reg = HTSRegressor(model='prophet', revision_method='OLS')
# reg = reg.fit(df=hsd, nodes=hier)
# preds = reg.predict(steps_ahead=10)