# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:34:53 2022

@author: Jesper Holm
"""

import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')
from pathlib import Path


def csv_gen(mode,rate):
    
    
    if mode == 'Both':
        
        Time = np.array([250,2000,2000+rate,4000])
        Temp = np.array([232.05,232.05,243,243])
    
        a = -21.492
        b = 0.0811
        Bdot = np.exp(a+b*Temp)
    elif mode == 'Temp':
        Time = np.array([250,2000,2000+rate,4000])
        Temp = np.array([232.05,232.05,243,243])
        Bdot = np.full(len(Time),1.9e-1)
    elif mode == 'Acc':
        Time = np.array([250,2000,2000+rate,4000])
        Temp = np.full(len(Time),232.05)
        Bdot = np.array([0.185,0.185,0.26,0.26])

        #plt.plot(Time,Temp)
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    
    np.savetxt('CFM/CFM_main/CFMinput/Equi/'+str(mode)+'/' + str(rate) + 'y/Acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/Equi/'+str(mode)+'/' + str(rate) + 'y/Temp.csv',Temp_csv,delimiter=',')
    
Modes = np.array(['Temp','Acc','Both'])
Rates = np.array([50,200,500,1000])
for j in range(len(Modes)):
    for i in range(len(Rates)):
        path = Path('CFM/CFM_main/CFMinput/Equi/' + str(Modes[j]) + '/'+ str(Rates[i])+ 'y')
        path.mkdir(parents=True, exist_ok=True)
        csv_gen(Modes[j],Rates[i])