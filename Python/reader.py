# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:15:36 2022

@author: jespe
"""

import h5py as h5
import os
import numpy as np


def read(rfolder):
    '''
    Parameters
    ----------
    rfoler : filepath
        filepath of savefolder.
    saver : Boolean
        True for figure saving.

    Returns
    -------
    Various arrays.

    '''
    rfile = 'Temp_HLdynamic1.1.hdf5'
    fn = os.path.join(rfolder,rfile)
    f = h5.File(fn,'r')
    
    z = f['depth'][:]
    climate = f["Modelclimate"][:]
    model_time = np.array(([a[0] for a in z[:]]))
    close_off_depth = f["BCO"][:, 2]
    close_off_age = f["BCO"][:,1]
    age_dist = f['gas_age'][:]
    #### COD variables
    temperature = f['temperature'][:]
    temp_cod = np.ones_like(close_off_depth)
    
    #age = f['age'][:]
    #delta_age = np.ones_like(close_off_depth)
    #ice_age_cod = np.ones_like(close_off_depth)
    
    d15N = f['d15N2'][:]-1
    d15N_cod = np.ones_like(close_off_depth)
    #d15n_grav = f1['d15N2'][:]-1
    d15n_grav_cod = np.ones_like(close_off_depth)
    
    #d40Ar = f['d40Ar'][:]-1
    #d40Ar_cod = np.ones_like(close_off_depth)
    #d40ar_grav = f1['d40Ar'][:]-1
    #d40ar_grav_cod = np.ones_like(close_off_depth)
    
    #gas_age = f['gas_age'][:]
    #gas_age_cod = np.ones_like(close_off_depth)
    for i in range(z.shape[0]):
        idx = int(np.where(z[i, 1:] == close_off_depth[i])[0])
        d15N_cod[i] = d15N[i,idx]
        #d15n_grav_cod[i] = d15n_grav[i,idx]
     #   d40Ar_cod[i] = d40Ar[i,idx]
        #d40ar_grav_cod[i] = d40ar_grav[i,idx]
        
        temp_cod[i] = temperature[i,idx]
      #  ice_age_cod[i] = age[i,idx]# - close_off_age[i]
       # delta_age[i] = age[i, idx] - gas_age[i, idx]
        #gas_age_cod[i] = close_off_age[i]-ice_age_cod[i]
        
        
        #print(ice_age_cod.shape)

    delta_temp = climate[:,2] - temp_cod
    d15N_th_cod = d15N_cod - d15n_grav_cod
    #d40Ar_th_cod = d40Ar_cod - d40ar_grav_cod
        
    
    
    
    
    
    f.close()
    with h5.File(fn,'r') as hf:
        print(hf.keys())
    return model_time,z,temperature,climate,d15N,close_off_depth,age_dist


def find_constant_row(matrix, tolerance=1e-6):
  for i, row in enumerate(matrix):
    if all(abs(x - row[0]) < tolerance for x in row):
      return i
  return -1

def find_first_constant(vector, tolerance):
  for i in range(1, len(vector)):
    if abs(vector[i] - vector[i-1]) <= tolerance:
      return i 
  return -1

'''
timesteps,depth,temperature,age,climate,d15N2,d40Ar,Bubble = read('CFM/CFM_main/CFMoutput/Equi/Both/HLdynamic/50y/')
from matplotlib import pyplot as plt
fig,ax = plt.subplots(1)
ax.invert_yaxis()

ax.plot(timesteps,Bubble)
Time_Const = timesteps[find_first_constant(Bubble[510:], tolerance=1e-4)+510]
ax.axvline(Time_Const,color='k',label=str(Time_Const))
Time = timesteps[find_constant_row(temperature[510:,1:],tolerance = 1e-1)]
ax.legend
'''


timesteps,depth,temperature,climate,d15N2,Bubble,age_dist = read('CFM/CFM_main/CFMoutput/EquiAmp2/Temp/HLdynamic/1.1')




def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w