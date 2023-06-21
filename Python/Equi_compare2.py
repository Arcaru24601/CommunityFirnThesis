# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:23:56 2023

@author: Jesper Holm
"""

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
#import reader as re
sns.set()
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, 28)
from pathlib import Path
import math
import pandas as pd



plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


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
    
def get_std_time(array):
        # print(value)
    # print(np.mean(array))
    # print(value)
    idx = (np.abs(array - 0.006)).argmin()
    # print(idx)
    return idx

class CoD_plotter():

    def __init__(self,j,i,filepath=None,filepath_2=None,rate=None,Exs=None,KtC=False):
        self.filepath = filepath
        self.filepath_2 = filepath_2
        self.KtC = KtC
        #if Exs == 'Acc/':
        #    fig, ax = plt.subplots(3, sharex=False, sharey=False)
        #else:
        

        self.Rates = np.array([float(x[:-1]) for x in rate])
        print(self.Rates)
        
        
        fig, ax = plt.subplots(6, sharex=True, sharey=False,constrained_layout=True)
        label = rate
        #fig.set_figheight(15)
        #fig.set_figwidth(8)

        #Rates = np.array([int(x[:-1]) for x in rate])
        #print(Rates)
        alpha = [1, 0.6, 0.3]
        alphas = [1,0.75,0.5,0.25]
        ax[2].invert_yaxis()
        ax[3].invert_yaxis()
        for k in range(len(self.filepath)):

            self.fpath = self.filepath[k]
            self.f2path = self.filepath_2[k]
            try:
                f = h5.File(self.fpath)
                f2 = h5.File(self.f2path)
            except Exception as e: print(e)

            #f = h5.File(self.fpath)
            #self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
            self.temp_cod = np.ones_like(self.close_off_depth)
            self.d15N = (f['d15N2'][:]-1)*1000
            self.d15N_g = (f2['d15N2'][:]-1)*1000

            d15N_cod = np.ones_like(self.close_off_depth)
            d15N_cod_g = np.ones_like(self.close_off_depth)
            for l in range(self.z.shape[0]):
                idx = int(np.where(self.z[l, 1:] == self.close_off_depth[l])[0])
                self.temp_cod[l] = self.temperature[l,idx]
                d15N_cod[l] = self.d15N[l,idx]
                d15N_cod_g[l] = self.d15N_g[l,idx]
                
            self.delta_temp = self.climate[:,2] - self.temp_cod
            
            
            
            
            #print(self.Rates,Rates)
            #print(self.Rates[k][:-1])
            slices = int(500+float(self.Rates[k]))
            
            
            T_h= self.climate[:, 2]
            T_c= self.temp_cod
            self.delta_temp= self.climate[:, 2] - self.temp_cod
            temp_mean = np.zeros(len(T_h))
            d15N_Temp = np.zeros(len(T_h))
            for h in range(len(T_h)):
                if not T_h[h] == T_c[h]:
                    temp_mean[h] = (T_h[h] * T_c[h]) / (T_h[h] - T_c[h]) * \
                                    np.log((T_h[h] / T_c[h]))
                    alpha= (8.856 - 1232/temp_mean[h]) * 10**(-3)
                    omega = alpha/temp_mean[h] * 10**(3)
                    
                    d15N_Temp[h] = ((T_h[h]/T_c[h])**alpha -1)*10**3#omega * self.delta_temp[h]
                else:
                    d15N_Temp[h] = 0
            
            
            
            
            
            
            
            d15N_Temp = d15N_cod-d15N_cod_g
            
            
            
            
            
            #print(label,rate)
            #print(len(self.filepath))        
            
            ax[0].plot(self.model_time, self.climate[:,2],color=cmap(cmap_intervals[k]))
            ax[0].grid(linestyle='--', color='gray', lw='0.5')
            ax[0].set_ylabel(r'Temp [K]')
            #ax[0].set_yticks(np.arange(230,260, step=10))
            #ax[0].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
            ax[1].plot(self.model_time, self.climate[:,1],color=cmap(cmap_intervals[k]))
            ax[1].grid(linestyle='--', color='gray', lw='0.5')
            ax[1].set_ylabel(r'Acc.\newline [$\mathrm{my}^{-1}$]')
            #ax[1].set_yticks(np.arange(0.05,0.6,step=0.2))
            #ax[1].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            
            ax[2].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[k]), label=label[k])
            ax[2].grid(linestyle='--', color='gray', lw='0.5')
            ax[2].set_ylabel(r'$z_{cod}$ [m]')
            #ax[2].set_yticks(np.arange(30,120,step=30))
            
            #get_HalfTime(self.close_off_depth)
            #print()
            slices = int((500+float(Rates[k][:-1])))
            #print(self.model_time[slices:])

            #Time_Const = self.model_time[find_first_constant(self.close_off_depth[510:], tolerance=1e-6)+510]
            Time_Const = self.model_time[get_HalfTime(self.close_off_depth[slices:],mode='Endpoint')+slices]
            #print(Time_Const)
            ax[2].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label=label[k]+'_'+str(Time_Const-1500-Rates[k]))
            #ax[2].legend(loc='lower right', fontsize=8)
            #ax[1].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            #ax[1].legend(loc='lower right', fontsize=8)
            #print(self.model_time[1000:])
            #box = ax[1].get_position()
                        #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            #ax[1].legend(ncol=1,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))


    
            #Times = np.array([0,Rates[k]+500,find_constant_row(self.temperature)])
            #print(Times)
            #if Exs == 'Acc/':
            #    continue
            #else:
            Time_Const = self.model_time[get_HalfTime(self.delta_temp[slices:],mode='Endpoint')+slices]
            ax[3].plot(self.model_time,self.delta_temp, color=cmap(cmap_intervals[k]))#, label=label[k])
            ax[3].axvline(x=Time_Const,color=cmap(cmap_intervals[k]),alpha=0.5)#,label=label[k]+'_'+str(Time_Const-1500-Rates[k]))
            ax[3].grid(linestyle='--', color='gray', lw='0.5')
            #ax[3].legend(loc='lower right', fontsize=8)

            ax[3].set_ylabel(r'$\Delta T$ [K]', labelpad=-1.5, fontsize=9)
            #ax[2].legend(loc='lower right', fontsize=8)
            #ax[4].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)

            Time_const_d15N = self.model_time[get_std_time(d15N_Temp[slices:])+slices]
            x = np.arange(0.3, 3.1, 0.1).round(2)
            ax[4].grid(linestyle='--', color='gray', lw='0.5')
            ax[4].plot(self.model_time, d15N_Temp,
                        color=cmap(cmap_intervals[k]))

            
            
            ax[4].axvline(x=Time_const_d15N, color=cmap(
                 cmap_intervals[k]), alpha=0.5)
            ax[4].set_ylabel(r"$\delta^{15}$N$_{th}$", labelpad=-1.5, fontsize=9)
            
            
            #ax[2].legend(loc='lower right', fontsize=8)
            ax[5].set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)

            Time_const_d15N_c = self.model_time[get_std_time(d15N_cod[slices:])+slices]
            x = np.arange(0.3, 3.1, 0.1).round(2)
            ax[5].grid(linestyle='--', color='gray', lw='0.5')

            ax[5].plot(self.model_time, d15N_cod,'--',
                        color=cmap(cmap_intervals[k]))
            
            ax[5].axvline(x=Time_const_d15N_c, color=cmap(
                 cmap_intervals[k]), alpha=0.5)
            ax[5].set_ylabel(r"$\delta^{15}$N", labelpad=-1.5, fontsize=9)
            
            title = ['a)','b)','c)','d)','e)','f)']

            ax[0].text(0, 0.9, title[0], size=11, ha='left', va='center',transform=ax[0].transAxes)
            ax[1].text(0, 0.9, title[1], size=11, ha='left', va='center',transform=ax[1].transAxes)
            ax[2].text(0, 0.9, title[2], size=11, ha='left', va='center',transform=ax[2].transAxes)
            ax[3].text(0, 0.9, title[3], size=11, ha='left', va='center',transform=ax[3].transAxes)
            ax[4].text(0, 0.9, title[4], size=11, ha='left', va='center',transform=ax[4].transAxes)
            ax[5].text(0, 0.9, title[5], size=11, ha='left', va='center',transform=ax[5].transAxes)

            #print(j,i)
            Expa = ['Temp/','Acc/','Both/']
            Modelas = ['HLdynamic/','Barnola1991/','Goujon2003/']
            
            path = Path('Test/'+ Expa[j] + Modelas[i])
            path.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame({'Model_time':self.model_time, 'temp':self.climate[:,2],'Acc':self.climate[:,1], 'delta_temp':self.delta_temp, 'CoD':self.close_off_depth})
            df.to_csv('Test/'+ Expa[j] + Modelas[i] + Rates[k] + '.csv')  
            
               
            #plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)

        f.close()
        #delta_temp, model_time, climate, close_off_depth

        
           
        
        
        
        return
    
    def Equi_output(self):

        odd = np.arange(1,6,2)
        even = np.arange(0,5,2)
        for k in range(len(self.filepath)):
            
            self.fpath = self.filepath[k]
            self.f2path = self.filepath_2[k]
            try:
                f = h5.File(self.fpath)
                f2 = h5.File(self.f2path)
            except Exception as e: print(e)
            #self.fpath.mkdir(parents=True, exist_ok=True)
            
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
            self.temp_cod = np.ones_like(self.close_off_depth)
            self.d15N = (f['d15N2'][:]-1)*1000
            self.d15N_g = (f2['d15N2'][:]-1)*1000

            d15N_cod = np.ones_like(self.close_off_depth)
            d15N_cod_g = np.ones_like(self.close_off_depth)
            for l in range(self.z.shape[0]):
                idx = int(np.where(self.z[l, 1:] == self.close_off_depth[l])[0])
                self.temp_cod[l] = self.temperature[l,idx]
                d15N_cod[l] = self.d15N[l,idx]
                d15N_cod_g[l] = self.d15N_g[l,idx]
                
            self.delta_temp = self.climate[:,2] - self.temp_cod
            
            
            
            
            #print(self.Rates,Rates)
            #print(self.Rates[k][:-1])
            slices = int(500+float(self.Rates[k]))
            
            
            T_h= self.climate[:, 2]
            T_c= self.temp_cod
            self.delta_temp= self.climate[:, 2] - self.temp_cod
            temp_mean = np.zeros(len(T_h))
            d15N_Temp = np.zeros(len(T_h))
            for h in range(len(T_h)):
                if not T_h[h] == T_c[h]:
                    temp_mean[h] = (T_h[h] * T_c[h]) / (T_h[h] - T_c[h]) * \
                                    np.log((T_h[h] / T_c[h]))
                    alpha= (8.856 - 1232/temp_mean[h]) * 10**(-3)
                    omega = alpha/temp_mean[h] * 10**(3)
                    
                    d15N_Temp[h] = ((T_h[h]/T_c[h])**alpha -1)*10**3#omega * self.delta_temp[h]
                else:
                    d15N_Temp[h] = 0
            
            
            
            
            
            
            
            d15N_Temp = d15N_cod-d15N_cod_g
            Time_Const_CoD = self.model_time[get_HalfTime(self.close_off_depth[slices:],mode='Endpoint')+slices]
            Time_Const_temp = self.model_time[get_HalfTime(self.delta_temp[slices:],mode='Endpoint')+slices]
            Time_const_d15N = self.model_time[get_std_time(d15N_Temp[slices:])+slices]
            Time_const_d15Nc = self.model_time[get_std_time(d15N_cod[slices:])+slices]

            
            
            Output[j*Num_int+k,odd[i]] = Time_Const_CoD - 1500 - int(self.Rates[k])    
            Output[j*Num_int+k,even[i]] = Time_Const_temp - 1500 - int(self.Rates[k])
            output_d15N[j*Num_int+k,i] = Time_const_d15N -1500 - int(self.Rates[k])
            output_d15Nc[j*Num_int+k,i] = Time_const_d15Nc -1500 - int(self.Rates[k])

        return Output[j*Num_int+0:j*Num_int+Num_int,even[i]:odd[i]+1], d15N_Temp, output_d15N[j*Num_int+0:j*Num_int+Num_int,i:i+1],output_d15Nc[j*Num_int+0:j*Num_int+Num_int,i:i+1]
rfolder = 'CFM/CFM_main/CFMoutput/Equi2/'
x = ['Temp','Acc','Both']

rfolder2 = 'CFM/CFM_main/CFMoutput/Equi2_grav/'

y = ['50y','200y','500y,','1000y','2000y']
x2 = ['HLD','BAR','GOU']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]




folder2 = './CFM/CFM_main/CFMinput/Equi2/Acc'
Equi_Folder = [name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))]

#folder2= './CFM/CFM_main/CFMinput/EquiAmp2/Acc'
y= [name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))]



Equi_Folder2 = np.asarray([float(x[:-1]) for x in Equi_Folder])
Equi_Folder2.sort()
Rates2 = [str(x) + 'y' for x in Equi_Folder2]

def folder_gen(Fold,Exp,FileFlag):
    X = [Exp]
    X2 = [Fold]
    Y = [x + '/' for x in Rates2] #### Move rate change to figure generation because of legend
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-4] + 'y' for x in Y]
        print(Y)
        #X2 = [s + '_' for s in X2]
        X = [s + '_' for s in X]
        Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    else:
        #X2 = [x[:-1] for x in X2]
        Y = [x[:-4] + 'y/' for x in Y]
        print(Y)
        Folder = [(i+i2+j) for i in X for i2 in X2 for j in Y]
    
    return Folder

Num_int= len(y)
S= Num_int*3
Matrix_15N = np.zeros((S,3))
output_d15N = np.zeros((S,3))

Matrix_15Nc = np.zeros((S,3))
output_d15Nc = np.zeros((S,3))


Output = np.zeros((S,6))    
Matrix = np.zeros((S,6))
even = np.arange(0,5,2)
odd = np.arange(1,6,2)    
Exp = ['Temp/','Acc/','Both/']


Rates = [str(x) + 'y' for x in Equi_Folder2]

Models = ['HLdynamic/','Barnola1991/','Goujon2003/']
#Rates = ['50y','200y','500y','1000y','2000y']
for j in range(len(Exp)):
    for i in range(len(Models)):
        T = folder_gen(Models[i],Exp[j],False)
        P = folder_gen(Models[i],Exp[j],True)
        path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
        path2 = [rfolder2 + m+n + '.hdf5' for m, n in zip(T, P)]

        print(Exp[j],Models[i])
        #print(path)
        Current_plot = CoD_plotter(j,i,filepath = path,filepath_2=path2,rate = Rates,Exs = Exp)
        plt.savefig('CoDEqui2/'+ str(Exp[j][:-1]) + str(Models[i][:-1]) +'.png',dpi=300)
        plt.close('all')
        Matrix[j*Num_int+0:j*Num_int+Num_int,even[i]:odd[i]+1], d15N_Temp,Matrix_15N[j*Num_int+0:j*Num_int+Num_int,i:i+1],Matrix_15Nc[j*Num_int+0:j*Num_int+Num_int,i:i+1] = Current_plot.Equi_output()
        Matrix[Num_int:Num_int*2,0:5:2] = 0


    #            

plt.close('all')


import pandas as pd
Models = ['HLD','Bar','GOU']
Output = ['Temps','CoD']

Iter1 = [Models,Output]
Exp = ['Temp','Acc','Both']
Iter2 = [Exp,Rates]
cols = pd.MultiIndex.from_product(Iter1)
    
idx = pd.MultiIndex.from_product(Iter2,names = ['Exp','dt'])


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


Rates1 = np.array([10,20,30,40,50,70,90])
Rates3 = np.linspace(100,2000,21,dtype=int)
Rates2 = np.concatenate((Rates1,Rates3))

x = Rates2
palette = sns.color_palette('Dark2', 3)
palette2 = sns.color_palette("Paired", 3)

linestyle = ['solid','dotted','dashed']

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='k',linestyle='solid', lw=4),
                Line2D([0], [0], color='k',linestyle='dotted', lw=4),
                Line2D([0], [0], color='k',linestyle='dashed', lw=4)]

Models = ['HLD', 'Bar', 'GOU']
Output= ['d15N']

Iter1= [Models, Output]
Exp = ['Temp', 'Acc', 'Both']
Multiplier= y
Iter2= [Exp, Multiplier]
cols= pd.MultiIndex.from_product(Iter1)

idx = pd.MultiIndex.from_product(Iter2, names = ['Exp', 'Amp'])


df2= pd.DataFrame(Matrix_15N,
                  columns=cols,
                  index=idx)

df2.style.set_table_styles(
    [dict(selector='th', props=[('text-align', 'center')])])
# df.set_properties(**{'text-align': 'center'})





df2.style.format(decimal='.', thousands=',', precision=1)
# df = df.astype(str)

df3= pd.DataFrame(Matrix_15Nc,
                  columns=cols,
                  index=idx)

df3.style.set_table_styles(
    [dict(selector='th', props=[('text-align', 'center')])])
# df.set_properties(**{'text-align': 'center'})





df3.style.format(decimal='.', thousands=',', precision=1)

fig, ax = plt.subplots(nrows = 3, ncols = 2,sharex=True,constrained_layout=True)
for k,exp in enumerate(['Temp', 'Acc', 'Both']):
    for i, model in enumerate(['HLD','Bar','GOU']):
        Array = df[str(model)].loc[str(exp)]
        Array2 = df2[str(model)].loc[str(exp)]
        #Array3 = df3[str(model)].loc[str(exp)]
        Temps = np.asarray(Array['Temps'])
        d15a = np.asarray(Array2['d15N'])
        #d15c = np.asarray(Array3['d15N'])

        CoD = np.asarray(Array['CoD'])
        ax[k,0].set_title(exp + r' change: $\Delta T$ ',fontsize=14)
        ax[k,1].set_title(exp + r' change: $z_{cod}$ ',fontsize=14)
        #ax[k,2].set_title(exp + ' d15N', fontsize=14)

        ax[k,0].plot(x,Temps,color=palette[k],linestyle=linestyle[i],label=model,lw=2)
        ax[k,1].plot(x,CoD,color=palette2[k],linestyle=linestyle[i],label=model,lw=2)
        #ax[k, 1].set_title(exp + ' Close-off depth', fontsize=14)

        #ax[k,2].plot(x, d15a, color=palette[k],
         #             linestyle=linestyle[i], label=model)
        fig.supylabel('Halfway-time to Equilibrium [y]',fontsize=14)
        fig.supxlabel('Duration of change [y]',fontsize=14)
        #fig.supxlabel('Duration of change [y]')
        #ax[k,1].set_xlabel('Duration of change [y]')
        #ax[k,0].set_ylabel('Temperature at close-off [K]')
        #ax[k,1].set_ylabel('Close-off depth [m]')
        
#box = ax[1,2].get_position()
            #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax[1,1].legend(custom_lines,['HLD','BAR','GOU'],ncol=1,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Equiplot/Dur2.png',dpi=300)














Rates1 = np.array([10,20,30,40,50,70,90])
Rates3 = np.linspace(100,2000,21,dtype=int)
Rates2 = np.concatenate((Rates1,Rates3))

x = Rates2
#x = np.arange(0.3, 3.1, 0.1).round(2)
fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex=True, constrained_layout=True)
for k, exp in enumerate(['Temp', 'Acc', 'Both']):
    for i, model in enumerate(['HLD', 'Bar', 'GOU']):
        Array= df2[str(model)].loc[str(exp)]
        Temps= np.asarray(Array['d15N'])
        #Array2= df3[str(model)].loc[str(exp)]
        #Temps2= np.asarray(Array2['d15N'])
        #CoD= np.asarray(Array['CoD'])
        ax[k].set_title(exp + ' change: d15N', fontsize=14)
        #ax[k, 1].set_title(exp + ' Close-off depth', fontsize=14)
        #ax2 = ax[k].twinx()
        ax[k].plot(x, Temps, color=palette[k],
                      linestyle=linestyle[i], label=model)
        #ax2.plot(x, Temps2, color=palette[k],
        #              linestyle=linestyle[i], label=model+'tot',alpha=0.5)
        #ax[k, 1].plot(x, CoD, color=palette2[k],
        #              linestyle=linestyle[i], label=model)
        fig.supylabel('Halfway-time to Equilibrium [y]', fontsize=14)
        fig.supxlabel('Duration of change [y]', fontsize=14)

# =============================================================================
#         Find way to have multiple common ylabels
# =============================================================================

box= ax[1].get_position()
            # ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax[1].legend(custom_lines, ['HLD', 'BAR', 'GOU'], ncol=1,
                fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
        # ax[k,0].set_ylabel('Temperature at close-off [K]')
        # ax[k,1].set_ylabel('Close-off depth [m]')
plt.savefig('Equiplot/Durd15n.png', dpi=300)







