# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:21:51 2023

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
#import reader as re
sns.set()


cmap_intervals = np.linspace(0, 1, 4)
from pathlib import Path
plt.close('all')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
    
class CoD_plotter():

    def __init__(self,j,filepath=None,KtC=False):
        self.filepath = filepath
        self.KtC = KtC

    def CoD_out(self):
            
        label = Models
        for i in range(len(self.filepath)):
            if not os.path.exists(self.filepath[i]):
                print('Results file does not exist', self.filepath[i][40:])
                continue
            f = h5.File(self.filepath[i])
            #print(self.filepath)
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            self.temperature = f['temperature'][:]
            if self.close_off_depth[i] < 0:
                continue
            
            self.d15N = f['d15N2'][:]-1
            self.d15N_cod = np.ones_like(self.close_off_depth)
            self.temp_cod = np.ones_like(self.close_off_depth)
            for k in range(self.z.shape[0]):
                idx = int(np.where(self.z[k, 1:] == self.close_off_depth[k])[0])
                self.d15N_cod[k] = self.d15N[k,idx]
                self.temp_cod[k] = self.temperature[k,idx]
            Grav = 9.81
            R = 8.3145
            delta_M = 1/1000  # Why is this 1/1000
            #print(T_means)
            self.d15N_cod_grav = (np.exp((delta_M*Grav*self.close_off_depth) / (R*self.temp_cod)) - 1) * 1000 
            #¤print(self.d15N_cod_grav)
            print(j,i)
            d15N_model[j,i] = self.d15N_cod_grav[-1]
            #print(self.close_off_depth.shape)
            CoD[j,i] = self.close_off_depth[-1]
               
        #plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        f.close()
        return d15N_model[j,:], CoD[j,:]
        
        
    
Models = ['HLdynamic/','HLSigfus/','Barnola1991/','Goujon2003/']    
CoD = np.zeros((10,4))
def folder_gen(Fold,FileFlag):
    X = [Fold]
    X2 = Models
    
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Folder = [(i2+i) for i in X for i2 in X2]
    else:
        #X2 = [s + '/' for s in X2]
        Folder = [(i2+i) for i in X for i2 in X2]
    
    return Folder 
 
os.chdir('../')
folder = 'CFM/CFM_main/CFMinput/Noise/'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
sub_folders = [x + '/' for x in sub_folders]


d15N_model = np.zeros((10,4))
Test = np.zeros((10,4))
CoD_T = np.zeros((10,4))
rfolder = 'CFM/CFM_main/CFMoutput/Noise/'

for j in range(len(sub_folders)):
    T = folder_gen(sub_folders[j],False)
    P = folder_gen(sub_folders[j],True)
    path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
    #print(path)
    Current_plot = CoD_plotter(j,path)
    Test[j,:],CoD_T[j,:]  = Current_plot.CoD_out()
    

Mod = ['HLD','HLS','BAR','GOU']

import pandas as pd

Temp_str = [x[:-1] for x in sub_folders]
df = pd.DataFrame(data = Test, 
                  index = sub_folders, 
                  columns = Mod)


linestyle = ['o','v','D','d']

Temps = [int(x[:-2]) for x in sub_folders] 
plt.figure(1)

for (index, column) in enumerate(df):
    print(df.columns[index])
    d15N = np.asarray(df[column])
    plt.plot(Temps,d15N,label=str(df.columns[index]),color=cmap(cmap_intervals[index]))
    plt.plot(Temps,d15N,linestyle[index],fillstyle='none',color=cmap(cmap_intervals[index]))

    
plt.legend()
