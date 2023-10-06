#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% Data
df = pd.read_parquet('bol/qs_predicted_sample.parquet')
#%%
from scipy import stats
df_ln1p = np.log1p(df.values.squeeze())
df_log2_1p = np.log2(1 + df.values.squeeze())
df_bc, bc_lambda = stats.boxcox(1 + df.values.squeeze())
df_log
#%%
fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = stats.probplot(df.values.squeeze(), dist=stats.lognorm(s=1), plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')
ax2 = fig.add_subplot(212)
prob = stats.probplot(df_bc, dist=stats.lognorm(s=1), plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')
plt.show()
#%%
x = stats.norm.rvs(size=50000)


res = stats.kstest(df.values.squeeze(), stats.lognorm(s=1).cdf)
res = stats.kstest(np.log1p(df_ln1p), stats.lognorm(s=0.57).cdf)