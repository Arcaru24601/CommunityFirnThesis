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
    rfile = 'CFMresults.hdf5'
    fn = os.path.join(rfolder,rfile)
    f = h5.File(fn,'r')
    
    z = f['depth'][:]
    climate = f["Modelclimate"][:]
    model_time = np.array(([a[0] for a in z[:]]))
    f_close_off_depth = f["BCO"][:, -1]
    close_off_depth = f["BCO"][:, 2]
    LID = f["BCO"][:, 6]
    close_off_age = f["BCO"][:,1]
    age_dist = f['gas_age'][:]
    #### COD variables
    temperature = f['temperature'][:]
    temp_cod = np.ones_like(close_off_depth)
    density = f['density'][:]
    #age = f['age'][:]
    #delta_age = np.ones_like(close_off_depth)
    #ice_age_cod = np.ones_like(close_off_depth)
    
    d15N = f['d15N2'][:]-1
    d15N_cod = np.ones_like(close_off_depth)
    #d15n_grav = f1['d15N2'][:]-1
    d15n_grav_cod = np.ones_like(close_off_depth)
    # Why is this 1/1000
    #print(T_means)
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
    return model_time,z,temperature,climate,d15N*1000,close_off_depth,age_dist,density,LID,f_close_off_depth


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


#timesteps,depth,temperature,climate,d15N2,Bubble,age_dist = read('CFM/CFM_main/CFMoutput/EquiAmp2/Temp/HLdynamic/1.1')
timesteps,depth,temperature,climate,d15N2,Bubble,age_dist,density,LiD,z_cod = read('CFM/CFM_main/CFMoutput/OptiNoise')


bcoMart =  1 / (1 / (917.0) + temperature[-1,-1] * 6.95E-7 - 4.3e-5)

s = 1 - (density[-1,1:] / 917.0)

def por_cl(s,bcos):
    s_co = 1 - bcoMart/917.0
    por_cl = 0.37 * s * np.power((s/s_co),-7.6)
    return por_cl


s_cl = por_cl(s, bcoMart)
s_op = s - s_cl


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
ind = s_cl>s
s_cl[ind] = s[ind]
s_op = s-s_cl
from matplotlib import pyplot as plt



fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)

ax[0].invert_yaxis()

ax[1].plot(s_cl[0:261],depth[-1,1:262],label=r'$s_cl$')
ax[1].plot(s_op[0:261],depth[-1,1:262],label=r'$s_op$')
ax[0].set_xlabel('Porosity')
ax[1].set_ylabel('Depth')
ax[0].set_xlabel('Density')
ax[0].axhline(LiD[-1],label='LiD')
ax[0].axhline(z_cod[-1],label='zCoD')
ax[1].axhline(z_cod[-1],label='zCoD')
ax[1].axhline(LiD[-1],label='LiD')
ax[1].axhline(Bubble[-1],label='CoD',color='k')
ax[0].axhline(Bubble[-1],label='CoD',color='k')

ax[0].plot(density[-1,1:262],depth[-1,1:262],label=r'$density$')
ax[2].set_xlabel(r'$\delta^{15}$N')
ax[2].plot(d15N2[-1,1:262],depth[-1,1:262])
ax[2].axhline(z_cod[-1],label='zCoD')
ax[2].axhline(LiD[-1],label='LiD')
ax[2].axhline(Bubble[-1],label='CoD',color='k')

plt.legend()





def read2(rfolder):
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
    rfile = 'CFMresults.hdf5'
    fn = os.path.join(rfolder,rfile)
    f = h5.File(fn,'r')
    
    z = f['depth'][:]
    climate = f["Modelclimate"][:]
    model_time = np.array(([a[0] for a in z[:]]))
    f_close_off_depth = f["BCO"][:, -1]
    close_off_depth = f["BCO"][:, 2]
    LID = f["BCO"][:, 6]
    close_off_age = f["BCO"][:,1]
    age_dist = f['gas_age'][:]
    #### COD variables
    temperature = f['temperature'][:]
    temp_cod = np.ones_like(close_off_depth)
    density = f['density'][:]
    
    d15N = f['d15N2'][:]-1

    diffusivity = f['diffusivity'][:]
    index = np.zeros(np.shape(diffusivity)[0])
    d15n_cod_diff = np.zeros(np.shape(diffusivity)[0])
    close_off_depth_diff = np.zeros(np.shape(diffusivity)[0])
    
    for i in range(np.shape(diffusivity)[0]):
        index[i] = np.max(np.where(diffusivity[i, 1:] > 10**(-20))) +1

        close_off_depth_diff[i] = z(i, int(index[i]))

        d15n_cod_diff[i] = d15N(i, int(index[i]))
    
    d15N_cod = np.ones_like(close_off_depth)
    #d15n_grav = f1['d15N2'][:]-1
    d15n_grav_cod = np.ones_like(close_off_depth)

    for i in range(z.shape[0]):
        idx = int(np.where(z[i, 1:] == close_off_depth[i])[0])
        d15N_cod[i] = d15N[i,idx]
 
        temp_cod[i] = temperature[i,idx]
   

    delta_temp = climate[:,2] - temp_cod
    d15N_th_cod = d15N_cod - d15n_grav_cod

    
    
    
    f.close()
    with h5.File(fn,'r') as hf:
        print(hf.keys())
    return model_time,z,temperature,climate,d15N*1000,close_off_depth,age_dist,density,LID,f_close_off_depth,close_off_depth_diff,d15n_cod_diff


timesteps,depth,temperature,climate,d15N2,Bubble,age_dist,density,LiD,z_cod,diff_cod,d15n_cod_diff = read('CFM/CFM_main/CFMoutput/OptiNoise')

