# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:57:17 2023

@author: jespe
"""
import numpy as np
import os
os.chdir('Optimization')
from Kindler_fit_ODR import expfunc, input_file
Point_N = np.array([0.549,0.662,0.462,0.276])
Point_T = np.array([213.811,219.22,235,250])
Point_A = np.array([0.0167,0.0535,0.1607,0.2621])


T,A,Beta = input_file(10)
Test2 = 245
Time = np.array([1000,1200,1400])
Temp = np.full_like(Time,Test2)
Bdot = np.full(len(Time),expfunc(Beta,Test2))
os.chdir('../')    
    
Temp_csv = np.array([Time,Temp])
Bdot_csv = np.array([Time,Bdot])
#print(Time,Temp,Bdot)
np.savetxt('CFM/CFM_main/CFMinput/Noise/Round4/optimize_acc.csv',Bdot_csv,delimiter=',')
np.savetxt('CFM/CFM_main/CFMinput/Noise/Round4/optimize_temp.csv',Temp_csv,delimiter=',')
    