# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:24:12 2023

@author: Jesper Holm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:09:45 2023

@author: Jesper Holm
"""

import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('plasma')
import seaborn as sns 
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#import reader as re
sns.set()
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
cmap = plt.cm.get_cmap('plasma')
cmap_intervals = np.linspace(0, 1, 5)
from pathlib import Path
import math

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

def find_last_constant(vector, tolerance):
    vector = np.flip(vector)    
    for i in range(1, len(vector)):
        if abs(vector[i] - vector[i-1]) <= tolerance:
            return i
        return -1


def get_HalfTime(array,mode):
    if mode == 'minmax':
        value = np.abs((np.max(array) + np.min(array)) / 2)
    elif mode == 'Endpoint':
        value = np.abs((array[0] + array[-1]) / 2)
        #print(value)
    #print(np.mean(array))
    #print(value)
    idx = (np.abs(array - value)).argmin()
    #print(idx)
    return idx
    
    
class CoD_plotter():

    def __init__(self,j,i,filepath=None,amp=None,Exs=None,KtC=False):
        self.filepath = filepath
        self.KtC = KtC
        #if Exs == 'Acc/':
        #    fig, ax = plt.subplots(3, sharex=False, sharey=False)
        #else:
        

            
        return
    
    def Equi_output(self):

        odd = np.arange(1,6,2)
        even = np.arange(0,5,2)
        for k in range(len(self.filepath)):
            
            self.fpath = self.filepath[k]
            f = h5.File(self.fpath)
            #self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
            self.temp_cod = np.ones_like(self.close_off_depth)

            for l in range(self.z.shape[0]):
                idx = int(np.where(self.z[l, 1:] == self.close_off_depth[l])[0])
                self.temp_cod[l] = self.temperature[l,idx]
                
                
                
            self.delta_temp = self.climate[:,2] - self.temp_cod
            
            
            
            
            
            
            slices = 500+300

            Time_Const_CoD = self.model_time[get_HalfTime(self.close_off_depth[slices:],mode='Endpoint')+slices]
            Time_Const_temp = self.model_time[get_HalfTime(self.delta_temp[slices:],mode='Endpoint')+slices]
            Output[j*5+k,odd[i]] = Time_Const_CoD - 1500 - 300    
            Output[j*5+k,even[i]] = Time_Const_temp - 1500 - 300
        return Output[j*5+0:j*5+5,even[i]:odd[i]+1]
rfolder = r'D:/CFMoutput/EquiAmp2/'
x = ['Temp','Acc','Both']

folder2 = './CFM/CFM_main/CFMinput/EquiAmp2/Acc'
y  = [name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))]
x2 = ['HLD','BAR','GOU']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]


def folder_gen(Fold,Exp,FileFlag):
    X = [Exp]
    X2 = [Fold]
    Y = [x + '/' for x in y] #### Move rate change to figure generation because of legend
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

Output = np.zeros((15,6))    
Matrix = np.zeros((15,6))
even = np.arange(0,5,2)
odd = np.arange(1,6,2)    
Exp = ['Temp/','Acc/','Both/']
Models = ['HLdynamic/','Barnola1991/','Goujon2003/']
Multiplier = [x + '/' for x in y]
for j in range(len(Exp)):
    for i in range(len(Models)):
        T = folder_gen(Models[i],Exp[j],False)
        P = folder_gen(Models[i],Exp[j],True)
        path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
        print(Exp[j],Models[i])
        #print(path)
        Current_plot = CoD_plotter(j,i,filepath = path,amp = Multiplier,Exs = Exp[j])
        Matrix[j*5+0:j*5+5,even[i]:odd[i]+1] = Current_plot.Equi_output()
        Matrix[5:10,0:5:2] = 0
        

    #            

plt.close('all')


import pandas as pd
Models = ['HLD','Bar','GOU']
Output = ['Temps','CoD']

Iter1 = [Models,Output]
Exp = ['Temp','Acc','Both']
Multiplier = y
Iter2 = [Exp,Multiplier]
cols = pd.MultiIndex.from_product(Iter1)
    
idx = pd.MultiIndex.from_product(Iter2,names = ['Exp','Amp'])


df = pd.DataFrame(Matrix,
                  columns = cols,
                  index = idx)

df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
#df.set_properties(**{'text-align': 'center'})

cell_hover = {
    "selector": "td:hover",
    "props": [("background-color", "#FFFFE0")]
}
index_names = {
    "selector": ".index_name",
    "props": "font-style: italic; color: darkgrey; font-weight:normal;"
}
headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
}




df.style.format(decimal='.', thousands=',', precision=1)
#df = df.astype(str)






fig, ax = plt.subplots(nrows = 3, ncols = 2)
for k,exp in enumerate(['Temp','Acc','Both']):
    for i, model in enumerate(['HLD','Bar','GOU']):
        Array = df[str(model)].loc[str(exp)]
        

        ax[k,0].plot(Array['Temps'])
        ax[k,1].plot(Array['CoD'])
        fig.supxlabel('Magnitude of change')
        
# =============================================================================
#         Find way to have multiple common ylabels
# =============================================================================
        
        
        #ax[k,0].set_ylabel('Temperature at close-off [K]')
        #ax[k,1].set_ylabel('Close-off depth [m]')
plt.savefig('Equiplot/Amp2.png',dpi=300)





