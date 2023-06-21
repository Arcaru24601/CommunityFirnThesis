# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:46:30 2023

@author: Jesper Holm
"""

from matplotlib import pyplot as plt
#from Kindler_fit_Clear import input_file, expfunc
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import seaborn as sns
sns.set()


Model_Temp = np.array([218.05,224.65,225.76,226.85,229.05,231.26,233.46,234.56,235.66,237.86,240.06,241.16,242.26,244.47])
Model_T_sig = np.array([[0.6,0.6,0.8,0.7],
                        [0.8,0.8,1.0,1.0],
                        [0.8,0.8,1.0,1.0],
                        [0.8,0.8,1.1,1.1],
                        [0.9,0.9,1.1,1.2],
                        [0.9,0.8,1.1,1.1],
                        [1,1.1,1.2,1.4],
                        [1.1,1,1.4,1.3],
                        [1.2,1.1,1.4,1.5],
                        [1.3,1.2,1.5,23.7],
                        [1.2,1.2,1.4,72.2],
                        [1.3,1.3,1.7,108.4],
                        [1.3,1.3,1.8,102.3],
                        [1.4,1.3,1.8,111]])


Model_T2 = np.array([220,225,235,240])
Model_S2 = np.array([[0.78,0.78,0.92,0.89],
                     [0.98,0.98,1.17,1.14],
                     [1.35,1.35,1.63,1.58],
                     [1.47,1.48,1.8,33.67]])




Model_T3 = np.array([220,225,230,235,240,245])
Model_S3 = np.array([[0.82,1.01,1.08],
                     [0.97,1.16,1.22],
                     [1.17,1.31,1.36],
                     [1.28,1.43,1.5],
                     [1.5,1.65,1.75],
                     [1.73,1.86,2.03]])





Model_T1 = np.array([218.05,220,224.65,225,225.76,226.85,229.05,231.26,233.46,234.56,235,235.66,237.86,240,240.06,241.16,242.26,244.47])
Model_S1 = np.array([[0.6,0.6,0.8,0.7],
                                                 [0.78,0.78,0.92,0.89],
                        [0.8,0.8,1.0,1.0],
                                                [0.98,0.98,1.17,1.14],
                        [0.8,0.8,1.0,1.0],
                        [0.8,0.8,1.1,1.1],
                        [0.9,0.9,1.1,1.2],
                        [0.9,0.8,1.1,1.1],
                        [1,1.1,1.2,1.4],
                        [1.1,1,1.4,1.3],
                        [1.35,1.35,1.63,1.58],
                        [1.2,1.1,1.4,1.5],
                        [1.3,1.2,1.5,23.7],
                        [1.47,1.48,1.8,33.67],
                        [1.2,1.2,1.4,72.2],
                        [1.3,1.3,1.7,108.4],
                        [1.3,1.3,1.8,102.3],
                        [1.4,1.3,1.8,111]])

plt.close('all')
Uncer = ['d15N','rho','Deff_o']
Model = ['HLD','HLS','BAR','GOU']
import seaborn as sns
palette = sns.color_palette(None,4)

linestyle = ['solid','dashed','dotted','dashdot']
fig,ax = plt.subplots(ncols=2,sharey=True,constrained_layout=True)
for i in range(len(Uncer)):
    print(i)
    ax[1].plot(Model_T3,Model_S3[:,i],label=Uncer[i])
for g in range(len(Model)):
    ax[0].plot(Model_Temp,Model_T_sig[:,g],'--',color=palette[g],label=Model[g])
    ax[0].plot(Model_T2,Model_S2[:,g],'-',color=palette[g])
    ax[0].set_ylim(0.6,2.1)
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
fig.supylabel(r'T$_{rec}$ uncertainty [K]',fontsize=14)
fig.supxlabel('Reference temperature [K]',fontsize=14)
ax[0].set_title('Densification models',fontsize=16)
ax[1].set_title('Parameters',fontsize=16)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].tick_params(axis='both', which='minor', labelsize=10)
plt.savefig('Depen_Temp.png', dpi=300)













