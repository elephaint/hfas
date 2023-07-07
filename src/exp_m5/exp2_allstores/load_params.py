#%%
import joblib
from pathlib import Path

current_path = Path(__file__).parent
param_dict = {}
add_path = 'lr0.1/_old'
# for experiment in experiments:
for path in current_path.joinpath(add_path).glob('*.params'):
    params = joblib.load(path)
    # params = optuna.load_study(storage=str(path), study_name=None)
    # print(path.stem)
    param_dict[path.stem] = params.best_trial.user_attrs['best_iter']
