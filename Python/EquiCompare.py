# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 00:02:43 2022

@author: Jesper Holm
"""


import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')
import seaborn as sns 
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
import reader as re
sns.set()
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, 4)
from pathlib import Path


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,filepath=None,KtC=False):
        self.filepath = filepath
        self.KtC = KtC
  
        fig, ax = plt.subplots(3, sharex=True, sharey=False)
        fig.set_figheight(15)
        fig.set_figwidth(8)
        Times = [500,1050,1100,1200,1500]
        alpha = [1, 0.75, 0.5, 0.25]
        label = ['50y', '100y', '200y','500y']
        for k in range(len(self.filepath)):
            self.fpath = self.filepath[k]
            f = h5.File(self.fpath)
            self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
           
            if self.KtC:
                self.climate[:,2] - 273.15

        
            
        
            
            ax[0].plot(self.model_time, self.climate[:,2],'k',color=cmap(cmap_intervals[k]))
            ax[0].grid(linestyle='--', color='gray', lw='0.5')
            ax[0].set_ylabel(r'\centering Temperature \newline\centering Forcing [K]')
            ax[0].set_yticks(np.arange(230,260, step=10))
            
            ax[1].plot(self.model_time, self.climate[:,1],'k',color=cmap(cmap_intervals[k]))
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'\centering Acc. Forcing \newline\centering [$\mathrm{my}^{-1}$ ice eq.]')
            ax[1].set_yticks(np.arange(0.05,0.6,step=0.2))
            
            ax[2].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[k]), label=label[k])
            ax[2].grid(linestyle='--', color='gray', lw='0.5')
            ax[2].set_ylabel(r'\centering Close-off \newline\centering depth [m]')
            ax[2].set_yticks(np.arange(30,120,step=30))
            ax[2].invert_yaxis()
            ax[2].legend(loc='lower right', fontsize=8)

            for i in range(len(Times)):

                ax[3].plot(self.temperature[Times[i]][:], self.z[Times[i]][:], color=cmap(cmap_intervals[k]), alpha=alpha[i], label=label[k])
                ax[3].grid(linestyle='--', color='gray', lw='0.5')
                if self.KtC:
                    ax[3].set_ylabel(r'\centering Temperature [\u00B0C]')
                else:
                    ax[3].set_ylabel(r"\centering Temperature [K]")
                ax[3].legend(loc='lower right', fontsize=8)
                
                
                
                
             
               
            plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
        
     
        f.close()
       
        return
    
  
            
rfolder = 'CFM/CFM_main/CFMoutput/Equi/'
x = ['Temp','Acc','Both']

y = ['50y','100y','200y,','500y']
x2 = ['HLD','BAR','GOU']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]


def folder_gen(Fold,FileFlag):
    X = ['Temp/','Acc/','Both/']
    X2 = Models
    Y = ['50y/','100y/','200y/','500y/']
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-1] for x in Y]

        #X2 = [s + '_' for s in X2]
        X = [s + '_' for s in X]
        Folder = [(i+i2) for i in X for i2 in X2]
    else:
        #X2 = [x[:-1] for x in X2]
        
        Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    
    return Folder 
Exp = ['Temp/','Acc/','Both/']
Models = ['HLdynamic/','Barnola1991/','Goujon2003/']
Rates = y
for j in range(len(Exp)):
    for i in range(len(Models)):
        T = folder_gen(Models[i],False)
        P = folder_gen(Models[i],True)
        path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
        
        Current_plot = CoD_plotter(filepath = path)
        #plt.savefig('CoDEqui/'+ str(Exp) + str(Models[i]) +'.png',dpi=300)
        #plt.close('all')

    #plt.close('all')            




