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


class CoD_plotter():

    def __init__(self,filepath=None,rate=None,Exs=None,KtC=False):
        self.filepath = filepath
        self.KtC = KtC
        #if Exs == 'Acc/':
        #    fig, ax = plt.subplots(3, sharex=False, sharey=False)
        #else:
        fig, ax = plt.subplots(4, sharex=False, sharey=False)
        label = rate
        fig.set_figheight(15)
        fig.set_figwidth(8)
        Rates = np.array([int(x[:-1]) for x in rate])
        alpha = [1, 0.6, 0.3]
        alphas = [1,0.75,0.5,0.25]
        ax[2].invert_yaxis()
        ax[3].invert_yaxis()
        for k in range(len(self.filepath)):
            if not os.path.exists(self.filepath[k]):
                print('Results file does not exist', self.filepath[k][40:])
                continue
            self.fpath = self.filepath[k]
            f = h5.File(self.fpath)
            #self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
           
            if self.KtC:
                self.climate[:,2] - 273.15

        
            #print(label,rate)
            #print(len(self.filepath))        
            ax[0].plot(self.model_time, self.climate[:,2],color=cmap(cmap_intervals[k]))
            ax[0].grid(linestyle='--', color='gray', lw='0.5')
            ax[0].set_ylabel(r'\centering Temperature \newline\centering Forcing [K]')
            #ax[0].set_yticks(np.arange(230,260, step=10))
            ax[0].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
            ax[1].plot(self.model_time, self.climate[:,1],color=cmap(cmap_intervals[k]))
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'\centering Acc. Forcing \newline\centering [$\mathrm{my}^{-1}$ ice eq.]')
            #ax[1].set_yticks(np.arange(0.05,0.6,step=0.2))
            ax[1].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
            ax[2].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[k]), label=label[k])
            ax[2].grid(linestyle='--', color='gray', lw='0.5')
            ax[2].set_ylabel(r'\centering Close-off \newline\centering depth [m]')
            #ax[2].set_yticks(np.arange(30,120,step=30))
            Time_Const = self.model_time[find_first_constant(self.close_off_depth[1010:], tolerance=1e-5)+1010]
            ax[2].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5,label=label[k]+str(Time_Const-2000))
            #ax[2].legend(loc='lower right', fontsize=8)
            ax[2].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            ax[2].legend(loc='lower right', fontsize=8)
            print(self.model_time[1000:])
            

    
            Times = np.array([0,Rates[k]+500,len(self.model_time)-1])
            #print(Times)
            #if Exs == 'Acc/':
            #    continue
            #else:
            for i in range(len(alpha)):
                
                ax[3].plot(self.temperature[Times[i]][1:], self.z[Times[i]][1:], color=cmap(cmap_intervals[k]), alpha=alpha[i], label=label[k]+str(Times[i]))
                ax[3].grid(linestyle='--', color='gray', lw='0.5')
                if self.KtC:
                    ax[3].set_ylabel(r'Depth')
                else:
                    ax[3].set_ylabel(r"Depth")
                #ax[3].legend(loc='lower right', fontsize=8)
                ax[3].set_xlabel(r"Temperature [K]", labelpad=-1.5, fontsize=9)
            ax[3].legend(loc='lower right', fontsize=8)

                
                
             
               
            #plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
        
        f.close()
       
        return
    
  
            
rfolder = 'CFM/CFM_main/CFMoutput/Equi/'
x = ['Temp','Acc','Both']

y = ['50y','200y','500y,','1000y']
x2 = ['HLD','BAR','GOU']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]


def folder_gen(Fold,Exp,FileFlag):
    X = [Exp]
    X2 = [Fold]
    Y = ['50y/','200y/','500y/','1000y/'] #### Move rate change to figure generation because of legend
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-1] for x in Y]

        #X2 = [s + '_' for s in X2]
        X = [s + '_' for s in X]
        Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    else:
        #X2 = [x[:-1] for x in X2]
        
        Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    
    return Folder 
Exp = ['Temp/','Acc/','Both/']
Models = ['HLdynamic/','Barnola1991/','Goujon2003/']
Rates = ['50y','200y','500y','1000y']
for j in range(len(Exp)):
    for i in range(len(Models)):
        T = folder_gen(Models[i],Exp[j],False)
        P = folder_gen(Models[i],Exp[j],True)
        path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
        print(Exp[j],Models[i])
        #print(path)
        Current_plot = CoD_plotter(filepath = path,rate = Rates,Exs = Exp[j])
        plt.savefig('CoDEqui/'+ str(Exp[j][:-1]) + str(Models[i][:-1]) +'.png',dpi=300)
        plt.close('all')

    plt.close('all')            




