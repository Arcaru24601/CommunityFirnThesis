# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:03 2022

@author: jespe
"""
from reader import read
import matplotlib.pyplot as plt 
from celluloid import Camera
plt.rcParams['font.size'] = '16'



def plotter(i,temperature,forcing,d15N2,depth,diffusivity,density,age):
        
    ax[0,0].plot(forcing[:,0],forcing[:,1])
    ax2=ax[0,0].twinx()
    ax2.plot(forcing[:,0],forcing[:,2])
    ax[0,0].set_xlabel(r'Model-time [yr]')
    ax[0,0].set_ylabel(r'Temperature forcing [K]')
    ax2.set_ylabel(r'Accumulation ice equivalent [m yr$^{-1}$]')
    
    ax[0,1].plot(d15N2[i,:],depth[:])
    ax[0,1].set_ylabel(r'Depth [m]')
    ax[0,1].set_xlabel(u'$\delta^{15}$N ‰')
    ax[0,1].xaxis.get_offset_text().set_visible(False)
    ax[0,1].set_ylim(0,121)
    ax[0,1].set_xlim(1000.0001,1000.2)
    ax[0,1].invert_yaxis()
    
    ax[0,2].plot(diffusivity[i,:],depth[:])
    ax[0,2].set_xlabel(r'Diffusivity [m$^2$ s$^{-1}$]')
    ax[0,2].set_ylabel(r'Depth [m]')
    ax[0,2].set_ylim(0,121)
    ax[0,2].set_xlim(-0.1e-5,2.3e-5)
    ax[0,2].invert_yaxis()
    
    ax[1,0].plot(density[i,1:],depth[:])
    ax[1,0].set_xlabel(r'Density [kg m$^{-3}$]')
    ax[1,0].set_ylabel(r'Depth [m]')
    ax[1,0].set_ylim(0,121)
    ax[1,0].set_xlim(300,1000)
    ax[1,0].invert_yaxis()
    
    ax[1,1].plot(temperature[i,1:],depth[:])
    ax[1,1].set_xlabel(r'Temperature [K]')
    ax[1,1].set_ylabel(r'Depth [m]')
    ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[1,1].set_xlim(243,255)
    ax[1,1].set_ylim(0,121)
    ax[1,1].invert_yaxis()
    
    ax[1,2].plot(age[i,1:],depth[:])
    ax[1,2].set_xlabel(r'Ice Age [yr]')
    ax[1,2].set_ylabel(r'Depth [m]')
    ax[1,2].set_ylim(0,121)
    ax[1,2].set_xlim(0,410)
    ax[1,2].invert_yaxis()
rfolder = 'CFM\CFM_main\CFMoutput_example\df'

timesteps,stps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,Bubble = read(rfolder)
#rows, cols = 2,3
#fig, ax = plt.subplots(rows,cols,figsize=(15, 15), tight_layout=True)
#plotter(-1,temperature,forcing,d15N2*1000,depth,diffusivity,density,age)


for i in range(stps):
    rows, cols = 2,3
    fig, ax = plt.subplots(rows,cols,figsize=(15, 15), tight_layout=True)
    print(i)
    plotter(i,temperature,forcing,d15N2*1000,depth,diffusivity,density,age)
    plt.savefig('ImageFolder/'+'{0:03d}'.format(i)+'.png')
    plt.close(fig)
    #plt.clf()

