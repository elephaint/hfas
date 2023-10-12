#%%
import numpy as np

# https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/preprocessing/_data.py#L3296
def yeo_johnson_transform(x, lambda_):
    xt = np.zeros_like(x)
    pos = x >= 0

    if lambda_ != 0:
        xt[pos] = (np.power(x[pos] + 1, lambda_) - 1) / lambda_
    else:
        xt[pos] = np.log1p(x[pos])
    
    if lambda_ != 2:
        xt[~pos] = - (np.power(-x[~pos] + 1, 2 - lambda_) - 1) / (2 - lambda_)
    else:
        xt[~pos] = -np.log1p(-x[~pos])

    return xt


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
def inverse_yeo_johnson_transform(xt, lambda_):
    x = np.zeros_like(xt)
    pos = xt >= 0

    if lambda_ != 0:
        x[pos] = np.power(xt[pos] * lambda_ + 1, 1 / lambda_) - 1
    else:
        x[pos] = np.exp(xt[pos]) - 1
    
    if lambda_ != 2:
        x[~pos] = 1 - np.power(-(2 - lambda_) * xt[~pos] + 1, 1 / (2 - lambda_))
    else:
        x[~pos] = 1 - np.exp(-xt[~pos])

    return x
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% Data
df = pd.read_parquet('bol/qs_predicted_sample.parquet')
#%%
lambda_ = 0.5
xb = df['quantitySold'].to_numpy().astype(np.float32)
xt = yeo_johnson_transform(xb, lambda_=lambda_)
x = inverse_yeo_johnson_transform(xt, lambda_=lambda_)
assert np.allclose(xb, x)
#%% Gradient and hessian
def yeo_johnson_transform_grad(x, lambda_):
    gradient = np.zeros_like(x)
    hessian = np.zeros_like(x)
    pos = x >= 0

    if lambda_ != 0:
        gradient[pos] = np.power(x[pos] + 1, lambda_ - 1)
        hessian[pos] = (lambda_ - 1) * np.power(x[pos] + 1, lambda_ - 2)
    else:
        gradient[pos] = 1 / (1 + x[pos])
        hessian[pos] = np.power(gradient, 2)
    
    if lambda_ != 2:
        gradient[~pos] = np.power(1 - x[~pos], 1 - lambda_)
        hessian[~pos] = (lambda_ - 1) * np.power(1 - x[~pos], -lambda_)
    else:
        gradient[~pos] = 1 / (1 - x[~pos])
        hessian[~pos] = np.power(gradient, 2)

    return gradient, hessian

lambda_ = 0.5
n_samples = 1
xb = df['quantitySold'].to_numpy().astype(np.float32)
gradient, hessian = yeo_johnson_transform_grad(xb, lambda_)
#%%
def yeo_johnson_transform_objective(preds, train_data):
    y = train_data.get_label()
    yhat = preds.astype(y.dtype)

    lambda_ = train_data.params['lambda_yj']
    gradient = np.zeros_like(yhat)
    hessian = np.zeros_like(yhat)
    pos = yhat >= 0
    if lambda_ != 0:
        gradient[pos] = np.power(yhat[pos] + 1, lambda_ - 1)
        hessian[pos] = (lambda_ - 1) * np.power(yhat[pos] + 1, lambda_ - 2)
    else:
        gradient[pos] = 1 / (1 + yhat[pos])
        hessian[pos] = np.power(gradient, 2)
    
    if lambda_ != 2:
        gradient[~pos] = np.power(1 - yhat[~pos], 1 - lambda_)
        hessian[~pos] = (lambda_ - 1) * np.power(1 - yhat[~pos], -lambda_)
    else:
        gradient[~pos] = 1 / (1 - yhat[~pos])
        hessian[~pos] = np.power(gradient, 2)

    gradient -= y

    return gradient, hessian

def l2_transformed(preds, eval_data):
    y = eval_data.get_label()
    yhatt = preds.astype(y.dtype)

    lambda_ = eval_data.params['lambda_yj']

    yhat =  yeo_johnson_transform(yhatt, lambda_)

    return 'l2_transformed', np.mean(np.square(yhat - y)), False
