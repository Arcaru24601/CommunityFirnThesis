# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 01:03:04 2023

@author: Jesper Holm
"""

import pandas as pd
df = pd.read_csv('TestAmp/Temp/HLdynamic/0.3.csv',sep=',')


from matplotlib import pyplot as plt

import numpy as np

def get_HalfTime(array,mode):
    if mode == 'minmax':
        value = np.abs((np.max(array) + np.min(array)) / 2)
    elif mode == 'Endpoint':
        #print(array[0],array[1])
        value = np.abs((array[0] + array[-1]) / 2)
        #print(value)
    #print(np.mean(array))
    #print(value)
    idx = (np.abs(array - value)).argmin()
    #print(idx)
    return idx


df = pd.read_csv('TestAmp/Temp/HLdynamic/0.3.csv',sep=',')
dt = np.asarray(df['delta_temp'])
CoD = np.asarray(df['CoD'])


Time_d = get_HalfTime(dt[800:], 'Endpoint')
Time_C = get_HalfTime(CoD[800:], 'Endpoint')

plt.figure()
plt.plot(df['Model_time'],df['delta_temp'])
plt.axvline(df['Model_time'][Time_d+800])


plt.figure()
plt.plot(df['Model_time'],df['CoD'])
plt.axvline(df['Model_time'][Time_C+800])