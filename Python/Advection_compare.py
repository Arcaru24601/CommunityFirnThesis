# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:52:11 2022

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

cmap_intervals = np.linspace(0, 1, 3)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,filepath=None,f1path=None,KtC=False):
        self.filepath = filepath
        self.f1path = f1path
        self.KtC = KtC
        rows,cols  = 3,2
        self.fig,self.ax = plt.subplots(rows,cols,figsize=(15, 8), sharex=True,sharey=False)
        
        return
 

    def Plotter(self):
        
        labels = ['Christo','Darcy','zero']
        
        for k in range(len(self.filepath)):
            if not os.path.exists(self.filepath[k]):
                print('Full results file does not exist')
                if not os.path.exists(self.f1path[k]):
                    print('Grav results file does not exist')
                continue
        
            f = h5.File(self.filepath[k])
            f1 = h5.File(self.f1path[k])
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            self.d15N = f['d15N2'][:]-1
            self.d15N_cod = np.ones_like(self.close_off_depth)
            self.d15n_grav = f1['d15N2'][:]-1
            self.d15n_grav_cod = np.ones_like(self.close_off_depth)
        
            self.d40Ar = f['d40Ar'][:]-1
            self.d40Ar_cod = np.ones_like(self.close_off_depth)
            self.d40ar_grav = f1['d40Ar'][:]-1
            self.d40ar_grav_cod = np.ones_like(self.close_off_depth)
        
            for i in range(self.z.shape[0]):
                idx = int(np.where(self.z[i, 1:] == self.close_off_depth[i])[0])
                self.d15N_cod[i] = self.d15N[i,idx]
                self.d15n_grav_cod[i] = self.d15n_grav[i,idx]
                self.d40Ar_cod[i] = self.d40Ar[i,idx]
                self.d40ar_grav_cod[i] = self.d40ar_grav[i,idx]
            
            self.d15N_th_cod = self.d15N_cod - self.d15n_grav_cod
            self.d40Ar_th_cod = self.d40Ar_cod - self.d40ar_grav_cod
            
            if self.KtC:
                self.delta_temp - 273.15
                self.climate[:,2] - 273.15
     
            f.close()
            self.ax[0,0].plot(self.model_time,self.d15N_cod * 1000,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            self.ax[1,0].plot(self.model_time,self.d15n_grav_cod * 1000,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            self.ax[2,0].plot(self.model_time,self.d15N_th_cod * 1000,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            self.ax[0,1].plot(self.model_time,self.d40Ar_cod * 1000/4,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            self.ax[1,1].plot(self.model_time,self.d40ar_grav_cod * 1000/4,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            self.ax[2,1].plot(self.model_time,self.d40Ar_th_cod * 1000/4,'--',label = labels[k])#,color=cmap(cmap_intervals[k]))
            
            
            self.ax[0,0].set_ylabel(r'$\delta^{15}N_{cod}$ [‰]')
            self.ax[1,0].set_ylabel(r'$\delta^{15}N_{cod,grav}$ [‰]')
            self.ax[2,0].set_ylabel(r'$\delta^{15}N_{cod,th}$ [‰]')
            self.ax[0,1].set_ylabel(r'$\delta^{40}Ar_{cod}$ [‰]')
            self.ax[1,1].set_ylabel(r'$\delta^{40}Ar_{cod,grav}$ [‰]')
            self.ax[2,1].set_ylabel(r'$\delta^{40}Ar_{cod,th}$ [‰]')
            for i in range(len(self.ax[:,0])):
                for j in range((len(self.ax[0,:]))):
                    self.ax[i,j].grid(linestyle='--', color='gray', lw='0.5')
                    self.ax[i,j].legend(loc='right', fontsize=8)

rfolder = 'CFM/CFM_main/CFMoutput/DO_event/'


def folder_gen(Model,exp,fold,FileFlag):
    X = [exp]
    X2 = [Model]
    Y = ['Christo/', 'Darcy/', 'zero/']
    Z = [fold]
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-1] for x in Y]
        Z = [x[:-1] for x in Z]
    Folder = [(i+i2+j+k) for i in X for i2 in X2 for j in Y for k in Z]
    return Folder 
exp = ['Long/', 'Short/', 'Osc2/','Osc/']
Models = ['HLdynamic/','Barnola1991/','Goujon2003/']
for i in range(len(Models)):
    for j in range(len(exp)):
        Fold_full = folder_gen(Models[i],exp[j],'full/',False)
        Fold_grav = folder_gen(Models[i],exp[j],'grav/',False)
        p_full = folder_gen(Models[i],exp[j],'full/',True)
        p_grav = folder_gen(Models[i],exp[j],'grav/',True)
        
        #path_grav = 'test'
        #path_full = 'test'
    
        path_grav = [rfolder + m+n + '.hdf5' for m,n in zip(Fold_grav,p_grav)]
        path_full = [rfolder + m+n + '.hdf5' for m,n in zip(Fold_full,p_full)]
    
    #print(path_grav)
        print(Models[i][:-1],exp[j][:-1])
        Current_plot = CoD_plotter(filepath=path_full,f1path=path_grav)
        Current_plot.Plotter()
        plt.savefig('Advection/'+str(Models[i][:-1]+exp[j][:-1])+'.png',dpi=300)
        plt.close('all')

plt.close('all')
