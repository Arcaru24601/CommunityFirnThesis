# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:20:44 2022

@author: jespe
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
cmap_intervals = np.linspace(0, 1, 7)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,f1path=None,f2path=None,f3path=None,KtC=False):
        path_list = [f1path, f2path, f3path]
        self.KtC = KtC
  
        
        for k in range(len(path_list)):
            self.fpath = path_list[k]
            f = h5.File(self.filepath)
            f1 = h5.File(self.f1path)

            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            self.close_off_age = f["BCO"][:,1]
            
            #### COD variables
            self.temperature = f['temperature'][:]
            self.temp_cod = np.ones_like(self.close_off_depth)
        
            self.age = f['age'][:]
            self.delta_age = np.ones_like(self.close_off_depth)
        
            self.d15N = f['d15N2'][:]-1
            self.d15N_cod = np.ones_like(self.close_off_depth)
            
            self.d40Ar = f['d40Ar'][:]-1
            self.d40Ar_cod = np.ones_like(self.close_off_depth)
            
            
            self.gas_age = f['gas_age'][:]
            self.gas_age_cod = np.ones_like(self.close_off_depth)
        
            for i in range(self.z.shape[0]):
                idx = int(np.where(self.z[i, 1:] == self.close_off_depth[i])[0])
                self.d15N_cod[i] = self.d15N[i,idx]
                self.d40Ar_cod[i] = self.d40Ar[i,idx]                
                self.temp_cod[i] = self.temperature[i,idx]
            
                self.delta_age[i] = self.age[i, idx] - self.gas_age[i, idx]
            self.delta_temp = self.climate[:,2] - self.temp_cod
            #self.delta_age[i] = self.age[i,idx] - self.close_off_age[i]
# =============================================================================
#             
# =============================================================================
            
        if self.KtC:
            self.delta_temp - 273.15
            self.climate[:,2] - 273.15
     
        f.close()
       
        return
    
    
    def plotting(self):
        fig, ax = plt.subplots(8, sharex=True, sharey=False)
        fig.set_figheight(15)
        fig.set_figwidth(8)
    
    
        alpha = [1, 0.6, 0.3]
        label = ['Barnola', 'HLdynamic', 'Goujon']
        for k in range(len(label)):

        
        
            ax[0].plot(self.model_time, self.climate[:,2],'k')
            ax[0].grid(linestyle='--', color='gray', lw='0.5')
            ax[0].set_ylabel(r'\centering Temperature \newline\centering Forcing [K]')
            ax[0].set_yticks(np.arange(230,260, step=10))
        
            ax[1].plot(self.model_time, self.climate[:,1],'k')
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'\centering Acc. Forcing \newline\centering [$\mathrm{my}^{-1}$ ice eq.]')
            ax[1].set_yticks(np.arange(0.05,0.6,step=0.2))
        
            ax[2].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            ax[2].grid(linestyle='--', color='gray', lw='0.5')
            ax[2].set_ylabel(r'\centering Close-off \newline\centering depth [m]')
            ax[2].set_yticks(np.arange(30,120,step=30))
            ax[2].invert_yaxis()
            ax[2].legend(loc='lower right', fontsize=8)

        
        
            ax[3].plot(self.model_time, self.temperature_cod, color=cmap(cmap_intervals[1]), alpha=alpha[k], label=label[k])
            ax[3].grid(linestyle='--', color='gray', lw='0.5')
            if self.KtC:
                ax[4].set_ylabel(r'\centering Temperature [\u00B0C]')
            else:
                ax[3].set_ylabel(r"\centering Temperature [K]")
            ax[3].legend(loc='lower right', fontsize=8)
            
            
            
            
            
            ax[4].plot(self.model_time, self.delta_temp, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            ax[4].grid(linestyle='--', color='gray', lw='0.5')
            if self.KtC:
                ax[4].set_ylabel(r'\centering Temperature \newline\centering Gradient [\u00B0C]')
            else:
                ax[4].set_ylabel(r'\centering Temperature \newline\centering Gradient [K]')
            ax[4].legend(loc='lower right', fontsize=8)

        
        
        
        
            ax[5].plot(self.model_time, self.close_off_age, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            ax[5].grid(linestyle='--', color='gray', lw='0.5')
            ax[5].set_ylabel(r'$\Delta$age [y]')
            ax[5].legend(loc='lower right', fontsize=8)

            
            
            ax[6].plot(self.model_time,self.d15N_cod * 1000, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            #ax[6].plot(self.model_time,self.d15N_th_cod * 1000,'c:',label='$\delta^{15}$N thermal')
            #ax[6].plot(self.model_time,self.d15n_grav_cod * 1000,'c--',label='$\delta^{15}$N gravitational')
            ax[6].grid(linestyle='--', color='gray', lw='0.5')
            ax[6].set_ylabel(r'$\delta^{15}N_{cod}$ [‰]')
            ax[6].set_yticks(np.arange(0.0,0.55, step=0.25))
            ax[6].legend(loc='lower right', fontsize=8)

        
        
            ax[7].plot(self.model_time,self.d40Ar_cod/4 * 1000, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            #ax[7].plot(self.model_time,self.d40Ar_th_cod/4 * 1000,'y:',label='$\delta^{40}$Ar thermal')
            #ax[7].plot(self.model_time,self.d40ar_grav_cod/4 * 1000,'y--',label='$\delta^{40}$Ar gravitational')
            ax[7].grid(linestyle='--', color='gray', lw='0.5')
            ax[7].set_ylabel(r'$\delta^{40}Ar_{cod}/4$ [‰]')
            ax[7].set_yticks(np.arange(0.0,0.55, step=0.25))
            ax[7].legend(loc='lower right', fontsize=8)
           
        plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        #plt.savefig('results/periodic_T_5000y.pdf')
        #plt.tight_layout
        #plt.show()
            
rfolder = 'CFM/CFM_main/CFMoutput/DO_event/'
x = ['Long', 'Short', 'Osc2','Osc']
x2 = ['HLdynamic','Barnola1991','Goujon2003']
y = ['zero', 'Christo', 'Darcy']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]


def folder_gen(fold,FileFlag):
    X = ['Long/', 'Short/', 'Osc2/','Osc/']
    X2 = [fold]
    Y = ['zero/', 'Christo/', 'Darcy/']
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-1] for x in Y]
    #Z = [fold]
    Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    return Folder 


Fold_HLD = folder_gen('HLdynamic/',False)
Fold_BAR = folder_gen('Barnola1991/',False)
Fold_GOU = folder_gen('Goujon2003/',False)

for i in range(len(Folder)):
    
    path_HLD = rfolder + Fold_HLD[i] + '/' + folder_gen('HLdynamic/',True)[i] + '.hdf5'
    path_BAR = rfolder + Fold_BAR[i] + '/' + folder_gen('Barnola1991/',True)[i] + '.hdf5'
    path_GOU = rfolder + Fold_GOU[i] + '/' + folder_gen('Goujon2003/',True)[i] + '.hdf5'
    print(Folder[i])
    if os.path.exists(path_HLD) and os.path.exists(path_GOU):
        Current_plot = CoD_plotter(f1path = path_HLD,f2path = path_BAR,f3path = path_GOU)
        Current_plot.plotting()
        plt.savefig('CoDv2/'+str(Folder[i])+'.png',dpi=300)
        plt.close('all')
    else:
        print("Results file does not exist")
plt.close('all')            
            
            