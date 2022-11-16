# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:17 2022

@author: jespe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:16:16 2022

@author: jespe
"""

import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')

period = 12
steps = 10000
Time= np.arange(1452,2021,1/12)
Temp = np.full_like(Time,242.05)# np.array([242.05,242.05,242.05])
Bdot = np.full(len(Time),0.19) 

#Time = np.array([0,30,10000])
#Temp = np.array([242.05,242.05,242.05])
#Bdot = np.array([0.19,0.19,0.19])


Temp_csv = np.array([Time,Temp])
Bdot_csv = np.array([Time,Bdot])

np.savetxt('CFM_2/CFM_main/CFMinput/Acc_const.csv',Bdot_csv,delimiter=',')
np.savetxt('CFM_2/CFM_main/CFMinput/Temp_const.csv',Temp_csv,delimiter=',')




