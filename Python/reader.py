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
    
    d15N = f['d15N2'][:]-1

    diffusivity = f['diffusivity'][:]
    index = np.zeros(np.shape(diffusivity)[0])
    d15n_cod_diff = np.zeros(np.shape(diffusivity)[0])
    close_off_depth_diff = np.zeros(np.shape(diffusivity)[0])
    
    for i in range(1,z.shape[0]):
        index = np.max(np.where(diffusivity[i, 1:] > 10**(-20))) +1
        #print(index,i)
        close_off_depth_diff[i] = z[i, int(index)]

        d15n_cod_diff[i] = d15N[i, int(index)]*1000
    
    d15N_cod = np.ones_like(close_off_depth)
    #d15n_grav = f1['d15N2'][:]-1
    d15n_grav_cod = np.ones_like(close_off_depth)

    for i in range(z.shape[0]):
        idx = int(np.where(z[i, 1:] == close_off_depth[i])[0])
        d15N_cod[i] = d15N[i,idx]*1000
 
        temp_cod[i] = temperature[i,idx]
   
    d15N_cod_z = np.ones_like(close_off_depth)

    for i in range(z.shape[0]):
        idx = int(np.where(z[i, 1:] == f_close_off_depth[i])[0])
        d15N_cod_z[i] = d15N[i,idx]*1000

        temp_cod[i] = temperature[i,idx]
  
    
   
    
   
    delta_temp = climate[:,2] - temp_cod
    d15N_th_cod = d15N_cod - d15n_grav_cod

    
    
    
    f.close()
    with h5.File(fn,'r') as hf:
        print(hf.keys())
    return model_time,z,temperature,climate,d15N*1000,close_off_depth,age_dist,density,LID,f_close_off_depth,close_off_depth_diff,d15n_cod_diff,diffusivity,d15N_cod_z,d15N_cod


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
timesteps,depth,temperature,climate,d15N2,Bubble,age_dist,density,LiD,z_cod,diff_cod,d15n_cod_diff,diffu,d15n_z,d15n_cod = read('CFM/CFM_main/CFMoutput/OptiNoise')


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

plt.close('all')

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)

ax[0].invert_yaxis()

ax[1].plot(s_cl[0:261],depth[-1,1:262],label=r'$s_cl$')
ax[1].plot(s_op[0:261],depth[-1,1:262],label=r'$s_op$')
ax[0].set_xlabel('Porosity')
ax[1].set_ylabel('Depth')
ax[0].set_xlabel('Density')
#ax[0].axhline(LiD[-1],label='LiD')
#ax[0].axhline(z_cod[-1],label='zCoD')
#ax[1].axhline(z_cod[-1],label='zCoD')
#ax[1].axhline(LiD[-1],label='LiD')
#ax[1].axhline(Bubble[-1],label='CoD',color='k')
#ax[0].axhline(Bubble[-1],label='CoD',color='k')

ax[0].plot(density[-1,1:262],depth[-1,1:262],label=r'$density$')
ax[2].set_xlabel(r'$\delta^{15}$N')
ax[2].plot(d15N2[-1,1:262],depth[-1,1:262])
ax[2].axhline(z_cod[-1],label='zCoD',color='g')
ax[2].axhline(LiD[-1],label='LiD')
ax[2].axhline(Bubble[-1],label='CoD',color='k')
ax[2].axhline(diff_cod[-1],label='diff CoD',color='r')


ax[0].axhline(z_cod[-1],label='zCoD',color='g')
ax[0].axhline(LiD[-1],label='LiD')
ax[0].axhline(Bubble[-1],label='CoD',color='k')
ax[0].axhline(diff_cod[-1],label='diff CoD',color='r')


ax[1].axhline(z_cod[-1],label='zCoD',color='g')
ax[1].axhline(LiD[-1],label='LiD')
ax[1].axhline(Bubble[-1],label='CoD',color='k')
ax[1].axhline(diff_cod[-1],label='diff CoD',color='r')







ax[2].axvline(d15n_cod[-1],linestyle=':')
ax[2].axvline(d15n_cod_diff[-1],linestyle='--')
#ax[2].axvline(d15n_z[-1],linestyle = '-.')


plt.legend()







def bco_rho(temp):
    bcoMart =  1 / (1 / (917.0) + temp * 6.95E-7 - 4.3e-5)
    return bcoMart

temp_test = np.linspace(215,250,10000)
bco = bco_rho(temp_test)






s = np.random.normal((bco[0]+bco[-1])/2,1.5*np.std(bco),size=800)
rhos = density[-1,1:100]/1000

fig,ax = plt.subplots(figsize=(5,9))
Z = 10*rhos
plt.plot(Z,rhos)
plt.axvline(6)
plt.axhline(0.6)
ax.invert_yaxis()


T = temp_test - 273.15
print((0.6-0.03)/0.00226-273.15)











