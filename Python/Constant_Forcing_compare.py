# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:02:23 2022

@author: jespe
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
        data['site_pressure'] = 691
    elif site == 'NEEM':
        data['site_pressure'] = 745
    elif site == 'DomeC':
        data['site_pressure'] = 658
    elif site == 'South':
        data['site_pressure'] = 681
    elif site == 'Summit':
        data['site_pressure'] = 665
    
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
        Bdot = np.full(len(Time),1.9e-1)

    elif site == 'DomeC':    
        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),247.75)
        Bdot = np.full(len(Time),1.3e-1)
    elif site == 'South':    
        Time = np.array([250,1000,5000])
        Temp = np.full(len(Time),254.2)
        Bdot = np.full(len(Time),1.2)
    elif site == 'Summit':
        Time = np.array([250,1000,5000])

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
Sites = ['NGRIP','Fuji','Siple','DE08','DML']


for Site in Sites:
    path = Path('CFM/CFM_main/CFMinput/Constant_Forcing/' + Site)
    path.mkdir(parents=True, exist_ok=True)
    csv_gen(Site)
    for Model in Models:
        print(Model,Site)
        Run(Model,Site)





