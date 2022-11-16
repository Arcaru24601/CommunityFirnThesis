# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:04:47 2022

@author: jespe
"""


import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')

period = 12
steps = 10000
Time =  np.array([1000,1500,10000])  #np.arange(1452,2021,1/12)
Temp = np.array([242.05,242.05,242.05])# np.array([242.05,242.05,242.05])
Bdot = np.array([0.19,0.19,0.19]) 

#Time = np.array([0,30,10000])
#Temp = np.array([242.05,242.05,242.05])
#Bdot = np.array([0.19,0.19,0.19])


Temp_csv = np.array([Time,Temp])
Bdot_csv = np.array([Time,Bdot])

np.savetxt('CFM_2/CFM_main/CFMinput/Acc_const2.csv',Bdot_csv,delimiter=',')
np.savetxt('CFM_2/CFM_main/CFMinput/Temp_const2.csv',Temp_csv,delimiter=',')

