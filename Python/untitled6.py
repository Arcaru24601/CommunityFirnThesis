# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:17:44 2023

@author: Jesper Holm
"""

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


Model = ['HLD','HLS','BAR','GOU']
title = ['a)','b)','c)']
plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,filepath=None,f1path=None,KtC=False):
        self.filepath = filepath
        #self.f1path = f1path
        self.KtC = KtC
        rows = 1
        cols = 3
        
        


        return
    
    
    def plotting(self):
        fig = plt.figure(figsize=(9, 8), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        size = 20

        ax0 = fig.add_subplot(spec[0, 0])
        #annotate_axes(ax0, 'ax0')
        
        ax10 = fig.add_subplot(spec[1, 0])
        #annotate_axes(ax10, 'ax10')6
        ax11 = fig.add_subplot(spec[:, 1])
        
        #self.fig,self.ax = plt.subplots(rows,cols, figsize=(10,5),constrained_layout=True)
        #f = h5.File(self.filepath)
       # f1 = h5.File(self.f1path)
        ax10.invert_yaxis()
        ax11.set_xlabel('Model time [years]',fontsize=size+2)
        ax0.set_xlabel('Model time [years]',fontsize=size+2)
        ax10.set_xlabel('Model time [years]',fontsize=size+2)
        ax10.grid(linestyle='--', color='gray', lw='0.5')
        ax10.set_ylabel(r'\centering Close-off depth [m]')
        ax11.grid(linestyle='--', color='gray', lw='0.5')
        ax11.set_ylabel(r'$\delta^{15}N_{cod}$ [‰]',fontsize=size+2)

        ax0.set_title(title[0],fontsize=size)
        ax10.set_title(title[1],fontsize=size)
        ax11.set_title(title[2],fontsize=size)
        
        
        
        ax0.tick_params(axis='both', which='major', labelsize=18)
        ax10.tick_params(axis='both', which='minor', labelsize=16)
        ax11.tick_params(axis='both', which='major', labelsize=18)
        ax0.tick_params(axis='both', which='minor', labelsize=16)
        ax10.tick_params(axis='both', which='major', labelsize=18)
        ax11.tick_params(axis='both', which='minor', labelsize=16)
        for i in range(len(self.filepath)):
            if not os.path.exists(self.filepath[i]):
                print('Results file does not exist', self.filepath[i][40:])
                continue
            f = h5.File(self.filepath[i])
        
            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            
            #### COD variables
            self.temperature = f['temperature'][:]
        
            self.d15N = f['d15N2'][:]-1
            self.d15N_cod = np.ones_like(self.close_off_depth)
        
            for g in range(self.z.shape[0]):
                idx = int(np.where(self.z[g, 1:] == self.close_off_depth[g])[0])
                self.d15N_cod[g] = self.d15N[g,idx]
            
     
            width = 1
        
        
            #labelFont = 15
            #legFont = 12
            #axFont = 13
            if i == 3:
                ax02 = ax0.twinx()
                ax02.tick_params(axis='both', which='major', labelsize=16)
                ax02.tick_params(axis='both', which='minor', labelsize=14)
                ax0.plot(self.model_time[251:-1], self.climate[251:-1,2],'g',linewidth=width)
                ax0.grid(linestyle='--', color='gray', lw='0.5')
                ax0.set_ylabel(r'Surface Temperature [K]',color='g',fontsize=size)
                ax02.set_ylabel("Accumulation Forcing [$\mathrm{my}^{-1}$ ice eq.]",color='y',fontsize=size)
                ax02.plot(self.model_time[251:-1], self.climate[251:-1,1],'y',linewidth=width)
                ax02.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
                ax02.set_ylim(0.18, 0.2)
                ax02.locator_params(axis='y', nbins=6)
            ax10.plot(self.model_time[251:-1], self.close_off_depth[251:-1],linewidth=width)
                        
            ax11.plot(self.model_time[251:-1],self.d15N_cod[251:-1] * 1000,'-',label=str(Model[i]),linewidth=width)

            ax11.legend(loc='best',fontsize=16)
            
            
            
            
        f.close()        
        

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



F_f = folder_gen('full',False)
P_f = folder_gen('full',True)
    
F_g = folder_gen('grav',False)
P_g = folder_gen('grav',True)
path = [rfolder + m + '/' + n + '.hdf5' for m,n in zip(F_f,P_f)]
path_grav = [rfolder + m + '/' + n + '.hdf5' for m,n in zip(F_g,P_g)]
    #path = rfolder + Fold_full[i] + '/' + folder_gen('full',True)[i] + '.hdf5'
    #path_grav = rfolder + Fold_grav[i] + '/' + folder_gen('grav',True)[i] + '.hdf5'
    #print(path_grav)
#print(Folder[i])
    #rows = 1
    #cols = 3
    #fig,ax = plt.subplots(rows,cols, figsize=(10,5),constrained_layout=True)
Current_plot = CoD_plotter(filepath=path,f1path=path_grav,KtC=True)
Current_plot.plotting()
        
        
        
        
plt.savefig('CoDv2/S_Combi.png',dpi=300)
plt.close('all')
2+2
plt.close('all')

