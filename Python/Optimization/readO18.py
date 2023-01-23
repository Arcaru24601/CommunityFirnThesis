# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:05:36 2023

@author: jespe
"""

import numpy as np
from matplotlib import pyplot as plt
Temp = None
Kindler_data = Temp
alpha = 0.30 #0.42
beta = 7
conv = -31.55
def dO18_convert(dO18,alpha,beta):
    Temp = (dO18 + 35.1)/(alpha) + conv + beta
    return Temp
import pandas as pd

file = 'Data.xlsx'
df = pd.read_excel(file, 'Ark1')

df2 = df.set_axis(['depth', 'ice age', 'gas age', 'temperature', 'temp error', 'accumulation', 'd18O', 'd18O age', 'age err', 'd15N', 'd15N err'], axis='columns', inplace=False)
df2['ice age'] *= -1
df2['gas age'] *= -1
df2['d18O age'] *= -1
df2 = df2[::-1]
Temp = dO18_convert(df2['d18O'], alpha, beta)


fig,ax = plt.subplots(3,sharex=True)
ax[0].plot(df2['ice age'],df2['temperature'])
ax[0].plot(df2['ice age'],Temp)
ax[1].plot(df2['ice age'],df2['accumulation'],'g')
ax[2].plot(df2['ice age'],df2['d18O'],'k')







