import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

path = 'CFMinput/'
filename = 'MERRA2_CLIM_df_72.5_-38.75'
with open(str(path) + str(filename) + '.pkl', 'rb') as file:
    object = pkl.load(file)

df = pd.DataFrame(object)
df.to_csv(r'%s%s.csv' % (path, filename))

# forcing
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('temperature and accumulation forcing')
ax1.plot(df['TSKIN'])
ax2.plot(df['BDOT'])
ax1.set(xlabel='time [yr]')
ax2.set(xlabel='time [yr]')
ax1.set(ylabel='temperature forcing [K]')
ax2.set(ylabel='accumulation forcing')
plt.tight_layout()
plt.show()