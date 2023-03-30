# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:31:59 2023

@author: jespe
"""

import numpy as np
import h5py as hf
def get_model_data(model_path):
    f = hf.File(model_path)
    
    z = f['depth'][:]
    climate = f["Modelclimate"][:]
    model_time = np.array(([a[0] for a in z[:]]))
    close_off_depth = f["BCO"][:,2]
    temperature = f['temperature'][:]
    #print(temperature)
    #print(close_off_depth)
    d15N = f['d15N2'][:]-1
    d15N_cod = np.ones_like(close_off_depth)
    f.close()
    #temp_cod = np.ones_like(close_off_depth)
    for k in range(z.shape[0]):
        idx = int(np.where(z[k, 1:] == close_off_depth[k])[0])
        d15N_cod[k] = d15N[k,idx]*1000
    Grav = 9.81
    R = 8.3145
    delta_M = 1/1000  # Why is this 1/1000
    #print(T_means)
    d15N_cod_grav = (np.exp((delta_M*Grav*close_off_depth) / (R*temperature[0,1])) - 1) * 1000 
    
    return d15N_cod_grav[-1], temperature[0,-1],d15N_cod[-1]#,close_off_depth,idx,z,d15N*1000,temperature

def get_model_data2(model_path):
    f = hf.File(model_path)
    
    z = f['depth'][:]
    climate = f["Modelclimate"][:]
    model_time = np.array(([a[0] for a in z[:]]))
    close_off_depth = f["BCO"][:,2]
    temperature = f['temperature'][:]
    f.close()
    temp_cod = np.ones_like(close_off_depth)
    for k in range(z.shape[0]):
        idx = int(np.where(z[k, 1:] == close_off_depth[k])[0])
        temp_cod[k] = temperature[k,idx]        
    Grav = 9.81
    R = 8.3145
    delta_M = 1/1000  # Why is this 1/1000
    #print(T_means)
    d15N_cod_grav = (np.exp((delta_M*Grav*close_off_depth) / (R*temp_cod)) - 1) * 1000 
    
    return d15N_cod_grav[-1], temperature[0,1],close_off_depth,model_time,z,climate,temperature


#Test = get_model_data2('../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5')


Tst = get_model_data2(r'C:/Users/jespe/Desktop/CFMoutput/OptiNoise/CFMresults.hdf5')


