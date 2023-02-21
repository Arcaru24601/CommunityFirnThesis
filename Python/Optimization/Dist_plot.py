# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:28:42 2023

@author: Jesper Holm
"""

        
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import glob
import h5py as h5py
for fcnt in range(1,4,1):
    fname = f'file_{fcnt}.h5'
    with h5py.File(fname,'w') as h5fw:
        arr = np.random.random(10*10).reshape(10,10)
        h5fw.create_dataset('data',data=arr)


cost_func = np.zeros(5)
d15N = np.zeros(5)
Acc = np.zeros(5)
Temp = np.zeros(5)


Models = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']
Dist = ['Dist_' + str(i) for i in range(4)]
# Loop over H5 files and load into a dataframe
for i in range(len(Dist)):
    fig, ax = plt.subplots(nrows=4,ncols=4)
    for j in range(len(Models)):
        for z,file in enumerate(glob.iglob('resultsFolder/HLdynamic/Dist_0/*.h5')):   
            with h5py.File(file, 'r') as h5fr:
            
                cost_func[z] = h5fr['cost_func'][-1]
            #d15N[z] = h5fr['d15N@CoD'][-1]
            #Acc[z] = h5fr['alpha'][-1]
            #Temp[z] = h5fr['ice_age'][-1]
        
        
        
        #table = pd.DataFrame(xdata).reset_index() 
        #print(table)
        
        ax[0,j].hist(cost_func,bins='auto') ### d15N input
        ax[1,j].hist(cost_func,bins='auto',color='r') ### Temperature output
        ax[2,j].hist(cost_func,bins='auto',color='g') ### Accumulation
        ax[3,j].plot(cost_func,color='k') ### Cost function