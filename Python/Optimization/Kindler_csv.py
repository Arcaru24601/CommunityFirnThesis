# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:58:37 2023

@author: jespe
"""

import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')
from pathlib import Path
from Kindler_fit import input_file


os.chdir('../')
import Test_script as je

Input_temp,Input_acc = input_file(num = 10)
#Input_temp_round = Input_temp.round()
def csv_gen(temp,bdot):
    
    
    
    Time = np.array([1000,2000,4000])
    Temp = np.full_like(Time,temp)
    Bdot = np.full(len(Time),bdot)
    
    
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    print(Time,Temp,Bdot)
    np.savetxt('CFM/CFM_main/CFMinput/Noise/'+str(int(temp)) + 'K/Acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/Noise/'+str(int(temp)) + 'K/Temp.csv',Temp_csv,delimiter=',')
    
#Modes = np.array(['Temp','Acc','Both'])
#Rates = np.array([50,200,500,1000])

for i in range(len(Input_temp)):
    path = Path('CFM/CFM_main/CFMinput/Noise/' + str(int(Input_temp[i])) + 'K')
    path.mkdir(parents=True, exist_ok=True)
    csv_gen(Input_temp[i],Input_acc[i])
        





Models = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']

def Equi_run(Model,temp,acc):
    for k in range(len(Model)):
        for i in range(len(temp)):
                print(Model[k],int(temp[i]))
                je.Terminal_run_Noise(Model[k],int(temp[i]),acc[i])




Equi_run(Models,Input_temp,Input_acc)







