#%%
import joblib
from pathlib import Path

current_path = Path(__file__).parent
param_dict = {}
# for experiment in experiments:
for path in current_path.glob('*.params'):
    params = joblib.load(path)
    # print(path.stem)
    # param_dict[path.stem] = params.best_trial.user_attrs['best_iter']
    param_dict[path.stem] = params
