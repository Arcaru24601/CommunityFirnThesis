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
        Time = np.array([250,1000,1500,1500+rate,5000])
        Temp = np.array([232.05,232.05,232.05,243,243])
        
        a = -21.492
        b = 0.0811
        Bdot = np.exp(a+b*Temp)
    elif mode == 'Temp':
        Time = np.array([250,1000,1500,1500+rate,5000])
        Temp = np.array([232.05,232.05,232.05,243,243])
        Bdot = np.full(len(Time),1.9e-1)
    elif mode == 'Acc':
        Time = np.array([250,1000,1500,1500+rate,5000])

        Temp = np.full(len(Time),232.05)
        Bdot = np.array([0.185,0.185,0.185,0.26,0.26])

        #plt.plot(Time,Temp)
        
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    
    np.savetxt('CFM/CFM_main/CFMinput/Equi2/'+str(mode)+'/' + str(rate) + 'y/Acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/Equi2/'+str(mode)+'/' + str(rate) + 'y/Temp.csv',Temp_csv,delimiter=',')

def csv_gen_amp(mode, multp):
    duration = 300
    amp_Temp = multp * 10
    amp_acc = multp * 0.075
    if mode == 'Both':
        Time = np.array([250,1000,1500,1500+duration,5000])
        Temp = np.array([232.05,232.05,232.05,232.05 + amp_Temp,232.05 + amp_Temp])
        
        a = -21.492
        b = 0.0811
        Bdot = np.exp(a+b*Temp)
        #plt.plot(Time,Temp)
        #plt.plot(Time,Bdot)

    elif mode == 'Temp':
        Time = np.array([250,1000,1500,1500+duration,5000])
        Temp = np.array([232.05,232.05,232.05,232.05+amp_Temp,232.05+amp_Temp])
        Bdot = np.full(len(Time),1.9e-1)
        #plt.figure()
        
    elif mode == 'Acc':
        Time = np.array([250,1000,1500,1500+duration,5000])

        Temp = np.full(len(Time),232.05)
        Bdot = np.array([0.185,0.185,0.185,0.185+amp_acc,0.185+amp_acc])
        #plt.figure()
        #plt.plot(Time,Bdot)
        
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    print(str(multp))
    np.savetxt('CFM/CFM_main/CFMinput/EquiAmp2/'+str(mode)+'/' + str(multp) + '/Acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/EquiAmp2/'+str(mode)+'/' + str(multp) + '/Temp.csv',Temp_csv,delimiter=',')

    
Modes = np.array(['Temp','Acc','Both'])
Rates = np.array([50,200,500,1000,2000])
Multiplier = np.array([0.3,1/2,1,2,3])
'''
for j in range(len(Modes)):
    for i in range(len(Rates)):
        path = Path('CFM/CFM_main/CFMinput/Equi/' + str(Modes[j]) + '/'+ str(Rates[i])+ '')
        path.mkdir(parents=True, exist_ok=True)
        csv_gen(Modes[j],Rates[i])
        path = Path('CFM/CFM_main/CFMinput/EquiAmp/' + str(Modes[j]) + '/'+ str(Multiplier[i])+ '')
        path.mkdir(parents=True, exist_ok=True)
        csv_gen_amp(Modes[j],Multiplier[i]) 
'''
   
Multiplier2 = np.linspace(start=0.3, stop=3, num=25)
Rates2 = np.linspace(start=10, stop=2000, num=25)        
for j in range(len(Modes)):
    for i in range(len(Rates2)):
        path = Path('CFM/CFM_main/CFMinput/Equi2/' + str(Modes[j]) + '/'+ str(Rates2[i])+ 'y')
        path.mkdir(parents=True, exist_ok=True)
        csv_gen(Modes[j],Rates2[i])
        path = Path('CFM/CFM_main/CFMinput/EquiAmp2/' + str(Modes[j]) + '/'+ str(Multiplier2[i])+ '')
        path.mkdir(parents=True, exist_ok=True)
        csv_gen_amp(Modes[j],Multiplier2[i])         
        
        
        
        