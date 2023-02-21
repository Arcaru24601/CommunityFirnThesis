# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:25:28 2023

@author: jespe
"""

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import glob
import matplotlib.ticker as mtick
import h5py as h5py
for fcnt in range(1,4,1):
    fname = f'file_{fcnt}.h5'
    with h5py.File(fname,'w') as h5fw:
        arr = np.random.random(10*10).reshape(10,10)
        h5fw.create_dataset('data',data=arr)


cost_func = np.zeros(50)
d15N = np.zeros(50)
count = np.zeros(50)
Temp = np.zeros(50)
np.random.seed(42)

#0.597, 0.463
#217.97, 229.96
Point_N = np.array([0.53,0.597,0.375,0.297])
Point_T = np.array([215.06,217.97,235,244.99])
#Point_A = np.array([0.0284,0.0535,0.1607,0.2621])

plt.close('all')
Models = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']
Dist = ['Dist' + str(i) for i in range(4)]

import seaborn as sns
sns.set_theme()
from astropy.visualization import hist
# Loop over H5 files and load into a dataframe
for i in range(len(Dist)):
    s = np.random.normal(Point_N[i],0.02,size=50)
    Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())][:50]

    fig, ax = plt.subplots(nrows=4,ncols=4,figsize=(8, 8), constrained_layout=True)
    for j in range(len(Models)):
        print('resultsFolder/Version1/' + str(Models[j]) + '/' + str(Dist[i]) + '/*.h5')
        for z,file in enumerate(glob.iglob('resultsFolder/Version1/' + str(Models[j]) + '/' + str(Dist[i]) + '/*.h5')):  
            with h5py.File(file, 'r') as h5fr:
                #print(h5fr.keys())
                #print(2)
                cost_func[z] = h5fr['cost_func'][-1]
                d15N[z] = h5fr['d15N@CoD'][-1]
                count[z] = h5fr['count'][-1]
                Temp[z] = h5fr['temp'][-1]
       
       
       
        #table = pd.DataFrame(xdata).reset_index()
        #print(table)
        bins = 'knuth'

        hist(Data_d15N, bins=bins, ax=ax[0,j],histtype='stepfilled',color='b')
        hist(Temp, bins=bins, ax=ax[1,j],histtype='stepfilled',color='r')
        hist(count, bins=bins, ax=ax[2,j],histtype='stepfilled',color='g')
        #hist(cost_func, bins=bins, ax=ax[3,j], histtype='stepfilled', density=True,color='k')
        #ax[0,j].hist(Data_d15N,bins='fd') ### d15N input
        #ax[1,j].hist(Temp,bins='fd',color='r') ### Temperature output
        #ax[2,j].hist(count,bins='fd',color='g') ### Accumulation
        ax[3,j].plot(cost_func,color='k') ### Cost function
        
        
        ax[0,j].set_xlabel(u'd15N [$\delta^{15}$N]')
        ax[1,j].set_xlabel('Temperature [K]')
        ax[2,j].set_xlabel('Point nr.')
        ax[3,j].set_xlabel('Point nr.')
        ax[2,j].set_ylabel('Iteration count')
        ax[3,j].set_ylabel('Cost function')
        ax[0,j].set_title(str(Models[j]))
        ax[3,j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        t = ax[3,j].yaxis.get_offset_text()
        t.set_x(1.1)