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
from Kindler_fit_Clear import input_file,expfunc,rho_0,rho_bco


os.chdir('..')
#print(os.getcwd())
#print(os.listdir())
import Test_script as je

Input_temp,Input_acc,Beta = input_file()

#Input_temp_round = Input_temp.round()
def csv_gen(temp,bdot):



    Time = np.array([1000,1500,2000])
    Temp = np.full(len(Time),temp)
    Bdot = np.full(len(Time),bdot)


    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    print(Time,Temp,Bdot)
    np.savetxt('CFM/CFM_main/CFMinput/Noise/Integer/'+str(int(temp)) + 'K/Acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/Noise/Integer/'+str(int(temp)) + 'K/Temp.csv',Temp_csv,delimiter=',')

#Modes = np.array(['Temp','Acc','Both'])
#Rates = np.array([50,200,500,1000])

for i in range(len(Input_temp)):
    path = Path('CFM/CFM_main/CFMinput/Noise/Integer/' + str(int(Input_temp[i])) + 'K')
    path.mkdir(parents=True, exist_ok=True)
    csv_gen(Input_temp[i],Input_acc[i])






Models = ['HLdynamic']#,'HLSigfus','Barnola1991','Goujon2003']
#Models = ['HLdynamic']
Diffus = ['Freitag', 'Schwander', 'Severinghaus', 'Witrant', 'Battle', 'Adolph']
def Equi_run(Model,temp,acc):
    for k in range(len(Model)):
        #if Model[k] == 'Goujon2003':
        #    Input_temp[-3] = 245
        #    Input_acc[-3] = expfunc(Beta,Input_temp[-2])
        for i in range(len(temp)):
                print(Model[k],int(temp[i]))
                je.Terminal_run_Noise(Model[k],int(temp[i]),acc[i])

def Diffu_run(Diffu,temp,acc):
    for k in range(len(Diffu)):
        #if Model[k] == 'Goujon2003':
        #    Input_temp[-3] = 245
        #    Input_acc[-3] = expfunc(Beta,Input_temp[-2])
        for i in range(len(temp)):
                print(Diffu[k],int(temp[i]))
                je.Terminal_run_Noise_Diffusivity(Diffu[k],int(temp[i]),acc[i])







def Equi_run_surf(Model,temp,acc):
    rho_s = rho_0(Input_temp, Input_acc)
    for k in range(len(Model)):
        #if Model[k] == 'Goujon2003':
        #    Input_temp[-3] = 245
        #    Input_acc[-3] = expfunc(Beta,Input_temp[-2])
        
        for i in range(len(temp)):
                
                print(Model[k],int(temp[i]))
                je.Terminal_run_Noise_rhos(Model[k],int(temp[i]),acc[i],rho_s[i])



















Equi_run(Models,Input_temp,Input_acc)
#Diffu_run(Diffus, Input_temp, Input_acc)
#Equi_run_surf(Models,Input_temp,Input_acc)


