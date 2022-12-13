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
from pathlib import Path

def osc(period):
    Time_steps = np.arange(0,4000)
    #T = 5000 ### Number of years
    F = 1/period
    amplitude = 10
    t_0 = 249
    Array = amplitude * np.sin(2*np.pi*F*Time_steps) + t_0
    Const = np.full(1000,t_0)
    Temp = np.concatenate((Const,Array))
    Time = np.arange(0,10000)
    #Bdot = np.full(len(Time),1.9e-1)
    a = -21.492
    b = 0.0811
    Bdot = np.exp(a+b*Temp)
    
    return Time, Temp, Bdot
def csv_gen(mode):
    if mode == 'Long':
        Time = np.array([250,1000,1050,3000,3500,5000])
        Temp = np.array([232.05,232.05,253,253,232.05,232.05])
        Bdot = np.full(len(Time),1.9e-1)
        #plt.plot(Time,Temp)
        #plt.plot(Time,Bdot)
        print(Bdot)
    elif mode == 'Peak':
        Time = np.array([250,1000,1050,1500,5000,5500,10000])
        Temp = np.array([232.05,232.05,253,232.05,232.05,232.05,232.05])
        Bdot = np.full(len(Time),1.9e-1)
        #plt.plot(Time,Temp)
        #plt.plot(Time,Bdot)
        print(Bdot)
    elif mode == 'Short':
        Time = np.array([250,1000,1050,1200,2000,4000,4050,5000,5500,7000,7050,7500,10000])
        Temp = np.array([232.05,232.05,253,253,232.05,232.05,253,253,232.05,232.05,253,232.05,232.05])
        Bdot = np.full(len(Time),1.9e-1)
        plt.plot(Time,Temp)
        #plt.plot(Time,Bdot)
        print(Bdot)
    elif mode == 'Osc':
        Time, Temp, Bdot = osc(5000)
        #plt.plot(Time,Temp)
    elif mode == 'Osc2':
        Time, Temp, Bdot = osc(500)
        #plt.plot(Time,Temp)
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    
    np.savetxt('CFM/CFM_main/CFMinput/DO_event/'+str(mode)+'/'+str(mode)+'_acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/DO_event/'+str(mode)+'/'+str(mode)+'_Temp.csv',Temp_csv,delimiter=',')
    
    
Events = np.array(['Long','Short','Osc2','Osc'])

for i in range(len(Events)):
    path = Path('CFM/CFM_main/CFMinput/DO_event/' + Events[i])
    path.mkdir(parents=True, exist_ok=True)
    csv_gen(Events[i])