# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:47:18 2022

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
#from scipy.signal import savgol_filter
#from scipy.ndimage.filters import uniform_filter1d
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker

plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,filepath=None,f1path=None,KtC=False):
        self.filepath = filepath
        self.f1path = f1path
        self.KtC = KtC
        
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
        self.ice_age_cod = np.ones_like(self.close_off_depth)
        
        self.d15N = f['d15N2'][:]-1
        self.d15N_cod = np.ones_like(self.close_off_depth)
        self.d15n_grav = f1['d15N2'][:]-1
        self.d15n_grav_cod = np.ones_like(self.close_off_depth)
        
        self.d40Ar = f['d40Ar'][:]-1
        self.d40Ar_cod = np.ones_like(self.close_off_depth)
        self.d40ar_grav = f1['d40Ar'][:]-1
        self.d40ar_grav_cod = np.ones_like(self.close_off_depth)
        
        self.gas_age = f['gas_age'][:]
        self.gas_age_cod = np.ones_like(self.close_off_depth)
        for i in range(self.z.shape[0]):
            idx = int(np.where(self.z[i, 1:] == self.close_off_depth[i])[0])
            self.d15N_cod[i] = self.d15N[i,idx]
            self.d15n_grav_cod[i] = self.d15n_grav[i,idx]
            self.d40Ar_cod[i] = self.d40Ar[i,idx]
            self.d40ar_grav_cod[i] = self.d40ar_grav[i,idx]
            
            self.temp_cod[i] = self.temperature[i,idx]
            self.ice_age_cod[i] = self.age[i,idx]# - self.close_off_age[i]
            self.delta_age[i] = self.age[i, idx] - self.gas_age[i, idx]
            self.gas_age_cod[i] = self.close_off_age[i]-self.ice_age_cod[i]
            
            
            #print(self.ice_age_cod.shape)

        self.delta_temp = self.climate[:,2] - self.temp_cod
        self.d15N_th_cod = self.d15N_cod - self.d15n_grav_cod
        self.d40Ar_th_cod = self.d40Ar_cod - self.d40ar_grav_cod
            
        if self.KtC:
            self.delta_temp - 273.15
            self.climate[:,2] - 273.15
     
        f.close()
       
        return
    
    
    def plotting(self):
        rows = 1
        title = ['a)','b)','c)']
        fig = plt.figure(figsize=(10, 8), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        
        ax0 = fig.add_subplot(spec[0, 0])
        #annotate_axes(ax0, 'ax0')
        
        ax10 = fig.add_subplot(spec[1, 0])
        #annotate_axes(ax10, 'ax10')6
        ax11 = fig.add_subplot(spec[:, 1])
        #annotate_axes(ax11, 'ax11')
        
        cols = 3
        #fig,ax = plt.subplots(rows,cols, figsize=(10,5),constrained_layout=True)
        #labelFont = 15
        #legFont = 12
        #axFont = 13
        ax0.tick_params(axis='both', which='major', labelsize=18)
        ax10.tick_params(axis='both', which='minor', labelsize=16)
        ax11.tick_params(axis='both', which='major', labelsize=18)
        ax0.tick_params(axis='both', which='minor', labelsize=16)
        ax10.tick_params(axis='both', which='major', labelsize=18)
        ax11.tick_params(axis='both', which='minor', labelsize=16)
        
        ax0.set_title(title[0],fontsize=20)
        ax10.set_title(title[1],fontsize=20)
        ax11.set_title(title[2],fontsize=20)
        
        
        ax02 = ax0.twinx()
        ax02.tick_params(axis='both', which='major', labelsize=18)
        ax02.tick_params(axis='both', which='minor', labelsize=16)
        
        ax0.plot(self.model_time[251:-1], self.climate[251:-1,2],'g')
        ax0.grid(linestyle='--', color='gray', lw='0.5')
        ax0.set_ylabel(r'Surface Temperature [K]',color='g',fontsize=22)
        ax02.set_ylabel("Accumulation Forcing [$\mathrm{my}^{-1}$ ice eq.]",color='y',fontsize=18)
        ax02.plot(self.model_time[251:-1], self.climate[251:-1,1],'y')
        print(self.d15N_cod)
        #ax0.set_yticks(np.arange(225,245, step=10))
        '''
        ax10.plot(self.model_time, self.climate[:,1],'k')
        ax10.grid(linestyle='--', color='gray', lw='0.5')
        ax10.set_ylabel(r'\centering Acc. Forcing \newline\centering [$\mathrm{my}^{-1}$ ice eq.]')
        ax10.set_yticks(np.arange(0.05,0.6,step=0.2))
        '''
        ax10.plot(self.model_time[251:-1], self.close_off_depth[251:-1],'b')
        ax10.grid(linestyle='--', color='gray', lw='0.5')
        ax10.set_ylabel(r'\centering Close-off depth [m]',fontsize=22)
        #ax10.set_yticks(np.arange(30,122,step=30))
        ax10.invert_yaxis()
        '''
        ax[3].plot(self.model_time, self.delta_temp,'r')
        ax[3].grid(linestyle='--', color='gray', lw='0.5')
        if self.KtC:
            ax[3].set_ylabel(r'\centering Temperature \newline\centering Gradient [\u00B0C]')
        else:
            ax[3].set_ylabel(r'\centering Temperature \newline\centering Gradient [K]')
        
        ax[4].plot(self.model_time, self.close_off_age,'g')
        #ax[4].plot(self.model_time, self.ice_age_cod,'g--')
        #ax[4].plot(self.model_time,self.gas_age_cod,'g')
        ax[4].grid(linestyle='--', color='gray', lw='0.5')
        ax[4].set_ylabel(r'$\Delta$age [y]')
        print(self.close_off_age[-10:]-self.ice_age_cod[-10:])
        '''
        ax11.plot(self.model_time[251:-1],self.d15N_cod[251:-1] * 1000,'c-',label='$\delta^{15}$N')
        ax11.plot(self.model_time[251:-1],self.d15N_th_cod[251:-1] * 1000,'c:',label='$\delta^{15}$N$_{th}$ ')
        ax11.plot(self.model_time[251:-1],self.d15n_grav_cod[251:-1] * 1000,'c--',label='$\delta^{15}$N$_{gr}$')
        ax11.grid(linestyle='--', color='gray', lw='0.5')
        ax11.set_ylabel(r'$\delta^{15}N_{cod}$ [‰]',fontsize=18)
        ax11.set_ylim(-0.1,0.6)
        #box = ax11.get_position()
                    #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                
        ax11.set_xlabel('Model time [years]',fontsize=22)
        ax0.set_xlabel('Model time [years]',fontsize=22)
        ax10.set_xlabel('Model time [years]',fontsize=22)
        # Put a legend to the right of the current axis
        ax11.legend(loc='upper right',fontsize=16)
        
        ax02.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax02.set_ylim(0.18, 0.2)
        ax02.locator_params(axis='y', nbins=6)
        #ax11.plot(self.model_time,self.d40Ar_cod/4 * 1000,'y-',label='$\delta^{40}$Ar')
        #ax11.plot(self.model_time,self.d40Ar_th_cod/4 * 1000,'y:',label='$\delta^{40}$Ar thermal')
        #ax11.plot(self.model_time,self.d40ar_grav_cod/4 * 1000,'y--',label='$\delta^{40}$Ar gravitational')
        #ax11.grid(linestyle='--', color='gray', lw='0.5')
        #ax11.set_ylabel(r'$\delta^{40}Ar_{cod}$ [‰]',fontsize=14)
        #ax11.set_yticks(np.arange(0.0,0.55, step=0.25))
        #ax11.legend(loc='right', fontsize=12)
        #ax11.set_ylabel('Model time [years]',fontsize=14)
        #box = ax11.get_position()
                    #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        #ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
        #ax11.set_xlabel('Model time [years]',fontsize=14)

rfolder = 'CFM/CFM_main/CFMoutput/DO_event/'
x = ['Long']
x2 = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']
y = ['Darcy']
#z = ['grav','Full']
Folder = [(i+i2+j) for i in x for i2 in x2 for j in y]


def folder_gen(fold,FileFlag):
    X = ['Long/']
    X2 = ['HLdynamic/','HLSigfus/','Barnola1991/','Goujon2003/']
    Y = ['Darcy/']
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Y = [x[:-1] for x in Y]
    Z = [fold]
    Folder = [(i+i2+j+k) for i in X for i2 in X2 for j in Y for k in Z]
    return Folder 


Fold_grav = folder_gen('grav',False)
Fold_full = folder_gen('full',False)


for i in range(len(Folder)):
    
    path = rfolder + Fold_full[i] + '/' + folder_gen('full',True)[i] + '.hdf5'
    path_grav = rfolder + Fold_grav[i] + '/' + folder_gen('grav',True)[i] + '.hdf5'
    #print(path_grav)
    if not os.path.exists(path):
        print('full results does not exist')
    elif not os.path.exists(path_grav):
        print('grav results does not exist')
        
        
    else:
        print(Folder[i])
        Current_plot = CoD_plotter(filepath=path,f1path=path_grav,KtC=True)
        Current_plot.plotting()
        
        
        
        
        plt.savefig('CoDv2/'+str(Folder[i])+'.png',dpi=300)
        plt.close('all')
    2+2
plt.close('all')

