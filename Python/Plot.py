# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:03 2022

@author: jespe
"""
from reader import read
import matplotlib.pyplot as plt 
from celluloid import Camera
plt.rcParams['font.size'] = '16'
cmap = plt.cm.get_cmap('viridis')
import numpy as np
import seaborn as sns
sns.set()
from pathlib import Path
import os
def plotter(i,ax,cm,temperature,forcing,d15N2,d40Ar,depth,diffusivity,density,age):
        
    ax[0,0].plot(forcing[:,0],forcing[:,1],'r-')
    ax2=ax[0,0].twinx()
    ax2.plot(forcing[:,0],forcing[:,2],'b-')
    ax[0,0].set_xlabel(r'Model-time [yr]')
    ax[0,0].set_ylabel(r'Temperature forcing [K]')
    ax2.set_ylabel(r'Accumulation ice equivalent [m yr$^{-1}$]')
    #print(depth.shape,d15N2.shape)
    
    
    
    
    ax10=ax[0,1].twiny()

    ax[0,1].set_xlabel("$\delta^{15}N$ [‰]",color=cmap(cmap_interval[3]))
    ax10.set_xlabel("$\delta^{40}Ar$ [‰]",color=cmap(cmap_interval[1]))
    #ax[0,1].set_xlabel("Model-time [yr]")
    
    ax10.set_xlim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[0,1].plot(d15N2[i,1:],depth[i,1:],color=cmap(cmap_interval[3]))
    ax10.plot(d40Ar[i,1:],depth[i,1:],color=cmap(cmap_interval[1]))
    ax[0,1].set_ylabel(r'Depth [m]')
    
    #ax[0,1].set_xlabel(u'$\delta^{15}$N ‰')
    ax[0,1].xaxis.get_offset_text().set_visible(False)
    #ax[0,1].set_ylim(0,121)
    #ax[0,1].set_xlim(1000.0001,1000.2)
    ax[0,1].invert_yaxis()
    
    ax[0,2].plot(diffusivity[i,1:],depth[i,1:],color=cmap(cm))
    ax[0,2].set_xlabel(r'Diffusivity [m$^2$ s$^{-1}$]')
    ax[0,2].set_ylabel(r'Depth [m]')
    #ax[0,2].set_ylim(0,121)
    #ax[0,2].set_xlim(-0.1e-5,2.3e-5)
    ax[0,2].invert_yaxis()
    
    ax[1,0].plot(density[i,1:],depth[i,1:],color=cmap(cm))
    ax[1,0].set_xlabel(r'Density [kg m$^{-3}$]')
    ax[1,0].set_ylabel(r'Depth [m]')
    #ax[1,0].set_ylim(0,121)
    #ax[1,0].set_xlim(300,1000)
    ax[1,0].invert_yaxis()
    
    ax[1,1].plot(temperature[i,1:],depth[i,1:],color=cmap(cm))
    ax[1,1].set_xlabel(r'Temperature [K]')
    ax[1,1].set_ylabel(r'Depth [m]')
    ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[1,1].set_xlim(243,255)
    #ax[1,1].set_ylim(0,121)
    ax[1,1].invert_yaxis()
    
    ax[1,2].plot(age[i,1:],depth[i,1:],color=cmap(cm))
    ax[1,2].set_xlabel(r'Ice Age [yr]')
    ax[1,2].set_ylabel(r'Depth [m]')
    #ax[1,2].set_ylim(0,121)
    #ax[1,2].set_xlim(0,410)
    ax[1,2].invert_yaxis()

folder = './CFM/CFM_main/CFMinput'
Folder = np.array(['df'])
#Folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
rfolder = 'CFM_2/CFM_main/CFMoutput_example/'

timesteps,stps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,d40Ar,Bubble = read(rfolder+Folder[0])

cmap_interval = np.linspace(0,1,7)
rows, cols = 2,3
fig, ax = plt.subplots(rows,cols,figsize=(15, 15), tight_layout=True)
plotter(-1,ax,cmap_interval[0],temperature,forcing,d15N2,d40Ar,depth,diffusivity,density,age)

rows,cols = 2,1

fig, axs = plt.subplots(2, 1, sharex=True)



axs[0].plot(timesteps,Bubble[:,2])
axs[1].plot(timesteps,Bubble[:,6])
axs[0].invert_yaxis()
axs[1].invert_yaxis()
#axs[1].set_ylim(130,60)
#axs[0].set_ylim(130,60)

# Remove vertical space between axes
print(Bubble[-1,2],Bubble[-1,6])
'''
for j in range(len(Folder)):
    timesteps,stps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,d40Ar,Bubble = read(rfolder+Folder[j])
    path = Path('CFM/CFM_main/CFMoutput/' + Folder[j])
    path.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(timesteps[::50])):
        rows, cols = 2,3
        fig, ax = plt.subplots(rows,cols,figsize=(15, 15), tight_layout=True)
        print(i)
        
        plotter(i,ax,cmap_interval[0],temperature,forcing,d15N2,d40Ar,depth,diffusivity,density,age)
        plt.savefig('ImageFolder/'+str(Folder[j])+'/{0:03d}'.format(i)+'.png')
        plt.close(fig)
    #plt.clf()

 '''
