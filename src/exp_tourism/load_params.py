#%%
import joblib
from pathlib import Path

current_path = Path(__file__).parent
best_trials = {}
best_feature_fractions = {}
params_dict =  {}
add_path = 'exp1_lr0.05'
# for experiment in experiments:
for path in current_path.joinpath(add_path).glob('*.params'):
    params = joblib.load(path)
    best_trials[path.stem] = params.best_trial.user_attrs['best_iter']
    params_dict[path.stem] = params
    best_feature_fractions[path.stem] = params.best_params['feature_fraction']
