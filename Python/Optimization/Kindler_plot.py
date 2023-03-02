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
            #print(self.filepath)
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            self.temperature = f['temperature'][:]
            if self.close_off_depth[i] < 0:
                continue
            #print(f.keys())
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
            d15N_model[j,i] = self.d15N_cod[-1]*1000#self.d15N_cod_grav[-1]
            d15N_firn[j,i] = self.d15N_cod[-1]
            #print(self.close_off_depth.shape)
            CoD[j,i] = self.close_off_depth[-1]
               
        #plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        f.close()
        return d15N_model[j,:], CoD[j,:], d15N_firn[j,:]
        
        
    
Models = ['HLdynamic/','HLSigfus/','Barnola1991/','Goujon2003/']    
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
folder = 'CFM/CFM_main/CFMinput/Noise/Round4/'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
sub_folders = [x + '/' for x in sub_folders]

H = len(sub_folders)
CoD = np.zeros((H,4))

d15N_model = np.zeros((H,4))
d15N_firn = np.zeros((H,4))
Test = np.zeros((H,4))
TestN = np.zeros((H,4))
CoD_T = np.zeros((H,4))
rfolder = 'CFM/CFM_main/CFMoutput/Noise/Round4/'

for j in range(len(sub_folders)):
    T = folder_gen(sub_folders[j],False)
    P = folder_gen(sub_folders[j],True)
    path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
    #print(path)
    print(sub_folders[j])
    Current_plot = CoD_plotter(j,path)
    Test[j,:],CoD_T[j,:],TestN[j,:]  = Current_plot.CoD_out()
    

Mod = ['HLD','HLS','BAR','GOU']

import pandas as pd

Temp_str = [x[:-1] for x in sub_folders]
df = pd.DataFrame(data = Test, 
                  index = sub_folders, 
                  columns = Mod)
df2 = pd.DataFrame(data = TestN, 
                  index = sub_folders, 
                  columns = Mod)


linestyle = ['o','v','D','d']

#Temps = [int(x[:-2]) for x in sub_folders] 
#plt.figure(1)

Point_N = np.array([0.3])
Point_T = np.array([245])
Point_A = np.array([0.2621])


os.chdir('Optimization')
from Kindler_fit_ODR import input_file,expfunc
Input_temp,Input_acc,Beta = input_file(num = 25)

Temps = Input_temp

plt.close('all')
fig = plt.figure(constrained_layout=True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
Point_N = np.array([0.53,0.597,0.375,0.297])
Point_T = np.array([215.06,217.97,235,244.99])
Point_A = np.array([0.0284,0.0535,0.1607,0.2621])

#df['GOU'][-1] = df['BAR'][-1]-0.025
for (index, column) in enumerate(df):
    print(df.columns[index])
    d15N = np.asarray(df[column])
    d15N_f = np.asarray(df[column])
    ax1.plot(Temps,d15N,label=str(df.columns[index]),color=cmap(cmap_intervals[index]))
    ax1.plot(Temps,d15N,linestyle[index],fillstyle='none',color=cmap(cmap_intervals[index]))
    #ax2.plot(Input_acc,d15N,label=str(df2.columns[index]),color=cmap(cmap_intervals[index]))
    ax2.plot(Input_acc,np.ones_like(Input_acc),linestyle[index],fillstyle='none',color=cmap(cmap_intervals[index]))
    #ax1.plot(Point_T,Point_N,'ko',lw=3,label='Dist. point' if index == 3 else "")
    #ax1.axvline(Point_T[0],color='r',linestyle='--',label='Point 1' if index == 3 else '')
    #ax1.axvline(Point_T[3],color='r',linestyle=':',label='Point 4' if index == 3 else '')
ax1.set_ylim(0.2,0.615)
ax2.set_ylim(ax1.get_ylim())
    #ax2 = ax1.twiny()    
    #plt.ylim((0.2,0.6))
ax1.set_xlabel('Temperature [K]',fontsize=14)
ax2.set_xlabel(r'Accumulation rate [m yr$^{-1}$]',fontsize=14)
box = ax1.get_position()
            #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
ax1.set_ylabel(u'$\delta^{15}$N [\u2030]',fontsize=14)
plt.savefig('Noise/NoiseTemp22.png',dpi=300)

fig = plt.figure(constrained_layout=True)
ax1 = fig.add_subplot(111)
#ax2 
ax2 = ax1.twiny()

for (index, column) in enumerate(df):
    print(df.columns[index])
    d15N = np.asarray(df[column])
    d15N_f = np.asarray(df[column])
    ax2.plot(Temps,np.ones_like(Temps),label=str(df.columns[index]),color=cmap(cmap_intervals[index]))
    #ax2.plot(Temps,d15N,linestyle[index],fillstyle='none',color=cmap(cmap_intervals[index]))
    ax1.plot(Input_acc,d15N,label=str(df2.columns[index]),color=cmap(cmap_intervals[index]))
    ax1.plot(Input_acc,d15N,linestyle[index],fillstyle='none',color=cmap(cmap_intervals[index]))
#ax1.plot(Point_T,Point_N,'o')

ax1.set_ylim(0.2,0.615)
ax2.set_xlabel('Temperature [K]',fontsize=14)
ax1.set_xlabel(r'Accumulation rate [m yr$^{-1}$]',fontsize=14)
box = ax1.get_position()
            #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
ax1.set_ylabel(u'$\delta^{15}$N [\u2030]',fontsize=14)


    
plt.legend()
plt.savefig('Noise/NoiseAcc.png',dpi=300)