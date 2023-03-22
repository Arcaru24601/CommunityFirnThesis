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
cmap_intervals = np.linspace(0, 1, 28)
from pathlib import Path
import math
import pandas as pd 
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
        fig, ax = plt.subplots(3, sharex=True, sharey=False,constrained_layout=True)
        label = amp
        #fig.set_figheight(15)
        #fig.set_figwidth(8)

        #Amplitude = np.array([int(x[:-1]) for x in amp])
        #print(Rates)
        alpha = [1, 0.6, 0.3]
        alphas = [1,0.75,0.5,0.25]
        ax[1].invert_yaxis()
        ax[2].invert_yaxis()
        for k in range(len(self.filepath)):
            '''
            if not os.path.exists(self.filepath[k]):
                print('Results file does not exist', self.filepath[k][40:])
                continue
            '''
            
                
            self.fpath = self.filepath[k]
            try:
                f = h5.File(self.fpath)
            except Exception as e: print(e)

            #self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
            self.temp_cod = np.ones_like(self.close_off_depth)

            for g in range(self.z.shape[0]):
                idx = int(np.where(self.z[g, 1:] == self.close_off_depth[g])[0])
                self.temp_cod[g] = self.temperature[g,idx]
                
                
                
            self.delta_temp = self.climate[:,2] - self.temp_cod
            #print(self.temp_cod.shape,self.climate[:,2].shape,self.delta_temp.shape)
            if self.KtC:
                self.climate[:,2] - 273.15

        
                    
            ax[0].plot(self.model_time, self.climate[:,2],color=cmap(cmap_intervals[k]))
            ax[0].grid(linestyle='--', color='gray', lw='0.5')
            ax[0].set_ylabel(r'\centering Temperature \newline\centering Forcing [K]')
            #ax[0].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            '''
            ax[1].plot(self.model_time, self.climate[:,1],color=cmap(cmap_intervals[k]))
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'\centering Acc. Forcing \newline\centering [$\mathrm{my}^{-1}$ ice eq.]')
            ax[1].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            '''
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'\centering Close-off \newline\centering depth [m]')
            
            slices = int((500+300))
            #print(slices)
            Time_Const = self.model_time[get_HalfTime(self.close_off_depth[slices:],mode='Endpoint')+slices]
            #print(self.model_time[400:])          
            print(Time_Const)
            if Exp[j] == 'Temp/' or Exp[j] == 'Both/':
                ax[1].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[k]), label=str(float(label[k][:-1])*10)+'K')
                ax[1].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label=str(float(label[k][:-1])*10)+'K_'+str(Time_Const-1500-300))
            elif Exp[j] == 'Acc/':
                ax[1].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[k]))#, label="{:0.3f}".format(float(label[k][:-1])*0.075))
                ax[1].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label="{:0.3f}".format(float(label[k][:-1])*0.075)+'_'+str(Time_Const-1500-300))
            
            ax[2].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            #ax[1].legend(loc='lower right', fontsize=8)
            

    
            Time_Const = self.model_time[get_HalfTime(self.delta_temp[slices:],mode='Endpoint')+slices]
            print(Time_Const)
            if Exp[j] == 'Temp/' or Exp[j] == 'Both/':
                ax[2].plot(self.model_time,self.delta_temp, color=cmap(cmap_intervals[k]))#, label=str(float(label[k][:-1])*10)+'K')    
                ax[2].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label=str(float(label[k][:-1])*10)+'K_'+str(Time_Const-1500-300))
            elif Exp[j] == 'Acc/':
                ax[2].plot(self.model_time,self.delta_temp, color=cmap(cmap_intervals[k]), label="{:0.3f}".format(float(label[k][:-1])*0.075))    
                ax[2].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label="{:0.3f}".format(float(label[k][:-1])*0.075)+'_'+str(Time_Const-1500-300))
            ax[2].grid(linestyle='--', color='gray', lw='0.5')

            ax[2].set_ylabel(r"\centering Temperature \newline \centering gradient [K]", labelpad=-1.5, fontsize=9)
            #box = ax[1].get_position()
                        #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            #ax[1].legend(ncol=1,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
            
            x = np.arange(0.3,3.1,0.1).round(2)

            #print(j,i)
            Expa = ['Temp/','Acc/','Both/']
            Modelas = ['HLdynamic/','Barnola1991/','Goujon2003/']
            
            path = Path('TestAmp/'+ Expa[j] + Modelas[i])
            path.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame({'Model_time':self.model_time, 'temp':self.climate[:,2],'Acc':self.climate[:,1], 'delta_temp':self.delta_temp, 'CoD':self.close_off_depth})
            df.to_csv('TestAmp/'+ Expa[j] + Modelas[i] + str(x[k]) + '.csv')                
                
             
               

        f.close()

            
        return
    
    def Equi_output(self):

        odd = np.arange(1,6,2)
        even = np.arange(0,5,2)
        for k in range(len(self.filepath)):
            
            self.fpath = self.filepath[k]
            
            try:
                f = h5.File(self.fpath)
            except Exception as e: print(e)

            #self.fpath.mkdir(parents=True, exist_ok=True)
            #print(self.fpath)
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
            
            f.close()
            
            
            
            
            
            
            slices = int((500+300))
            #print(self.model_time[800])
            Time_Const_CoD = self.model_time[get_HalfTime(self.close_off_depth[slices:],mode='Endpoint')+slices]
            Time_Const_temp = self.model_time[get_HalfTime(self.delta_temp[slices:],mode='Endpoint')+slices]
            Output[j*Num_int+k,odd[i]] = Time_Const_CoD - 1500 - 300    
            Output[j*Num_int+k,even[i]] = Time_Const_temp - 1500 - 300
        return Output[j*Num_int+0:j*Num_int+Num_int,even[i]:odd[i]+1]
rfolder = 'CFM/CFM_main/CFMoutput/EquiAmp2/'
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
Num_int = len(y)
S = Num_int*3


Output = np.zeros((S,6))    
Matrix = np.zeros((S,6))
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
        plt.savefig('CoDEquiAmp2/'+ str(Exp[j][:-1]) + str(Models[i][:-1]) +'.png',dpi=300)
        plt.close('all')
        try:
            Matrix[j*Num_int+0:j*Num_int+Num_int,even[i]:odd[i]+1] = Current_plot.Equi_output()
            #print(Matrix)
        except Exception as e: print(e)

        #Matrix[25:50,0:5:2] = 0
        

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





x = np.arange(0.3,3.1,0.1).round(2)
fig, ax = plt.subplots(nrows = 3, ncols = 2)
for k,exp in enumerate(['Temp','Acc','Both']):
    for i, model in enumerate(['HLD','Bar','GOU']):
        Array = df[str(model)].loc[str(exp)]
        Temps = np.asarray(Array['Temps'])
        CoD = np.asarray(Array['CoD'])


        ax[k,0].plot(x,Temps)
        ax[k,1].plot(x,CoD)
        fig.supxlabel('Magnitude of change')
        
# =============================================================================
#         Find way to have multiple common ylabels
# =============================================================================
        
        
        #ax[k,0].set_ylabel('Temperature at close-off [K]')
        #ax[k,1].set_ylabel('Close-off depth [m]')
plt.savefig('Equiplot/Amp2.png',dpi=300)





