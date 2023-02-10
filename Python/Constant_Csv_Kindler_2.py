# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:02:00 2023

@author: Jesper Holm
"""


import h5py as h5
import os,json,subprocess
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')
from pathlib import Path
#import Constant_Forcing_COD as CFD
def Run(Model,site):
    file = open('CFM/CFM_main/Constant_Forcing.json')
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['AutoSpinUpTime'] = False
    data['resultsFileName'] = str(Model) + str(site) + '.hdf5'
    data['resultsFolder'] = 'CFMoutput/Constant_Forcing/' + str(site) + '/' + str(Model)
    data['InputFileFolder'] = 'CFMinput/Constant_Forcing'
    if site == 'NGRIP':
        data['rhos0'] = 300
    elif site == 'NEEM':
        data['rhos0'] = 340
    elif site == 'GRIP':
        data['rhos0'] = 340
    elif site == 'DomeC':
        data['rhos0'] = 320
    elif site == 'South':
        data['rhos0'] = 350
    elif site == 'Summit':
        data['rhos0'] = 350          
    data['InputFileNamebdot'] = str(site) + '_acc.csv'
    data['InputFileNameTemp'] = str(site) + '_Temp.csv'
    data['physRho'] = str(Model)
        
    with open("CFM/CFM_main/Constant_Forcing.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    
    
    subprocess.run('python main.py Constant_Forcing.json -n', shell=True, cwd='CFM/CFM_main/')


def csv_gen(site):
    if site == 'NGRIP':    

        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),242.05)
        Bdot = np.full(len(Time),1.9e-1)
    elif site == 'NEEM':    
        Time = np.array([250,1000,5000])
        #Time = np.array([1,3000,5000])
        Temp = np.full(len(Time),242.05)
        Bdot = np.full(len(Time),2.2e-1)
    
    elif site == 'Summit':    
        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),241.75)
        Bdot = np.full(len(Time),2.3e-1)
    elif site == 'DomeC':    
        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),219.15)
        Bdot = np.full(len(Time),2.7e-2)
    elif site == 'South':
        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),222.15)
        Bdot = np.full(len(Time),8e-2)
    #plt.plot(Time,Temp)
    #plt.plot(Time,Bdot)
    print(Bdot)

        #plt.plot(Time,Temp)
    Temp_csv = np.array([Time,Temp])
    Bdot_csv = np.array([Time,Bdot])
    
    np.savetxt('CFM/CFM_main/CFMinput/Constant_Forcing/'+str(site)+'_acc.csv',Bdot_csv,delimiter=',')
    np.savetxt('CFM/CFM_main/CFMinput/Constant_Forcing/'+str(site)+'_Temp.csv',Temp_csv,delimiter=',')
    
    

file = open('CFM/CFM_main/Constant_Forcing.json')
data = json.load(file)
Models = data['physRho_options']
Sites = ['NGRIP','NEEM','Summit','DomeC','South']


for Site in Sites:
    path = Path('CFM/CFM_main/CFMinput/Constant_Forcing/' + Site)
    path.mkdir(parents=True, exist_ok=True)
    csv_gen(Site)
    for Model in Models:
        print(Model,Site)
        Run(Model,Site)