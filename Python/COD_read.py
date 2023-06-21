# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:16:16 2022

@author: jespe
"""

import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('viridis')
import seaborn as sns 
cmap = plt.cm.get_cmap('Dark2')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#import reader as re
sns.set()
from scipy.signal import savgol_filter
#from scipy.ndimage.filters import uniform_filter1d
from matplotlib import ticker

plt.close('all')
class CoD_plotter():
    

    def __init__(self,filepath=None):
        self.filepath = filepath
        #self.filepath = 'CFM/CFM_main/CFMoutput/Constant_Forcing/Fuji/KuipersMunneke2015'
        rfile = 'CFMresults.hdf5'
        #rfile = 'KuipersMunneke2015Fuji.hdf5'
        self.fs = os.path.join(self.filepath,rfile)
        f = h5.File(self.fs,'r')        
        #print(self.fs)
        #print(f.keys())
        
        self.z = f['depth'][:]
        self.rho = f['density'][:]
        self.temperature = f["temperature"][:]
        
        self.climate = f["Modelclimate"][:]
        self.model_time = np.array(([a[0] for a in self.z[:]]))
        self.close_off_depth = f["BCO"][:, 2]
        self.d15N = (f["d15N2"][:]-1.) * 1000
        #self.d40Ar = (f["d40Ar"][:]-1.) * 1000
        self.T_cod = np.ones_like(self.model_time)
        self.d15N_cod = np.ones_like(self.model_time)
        #self.d40Ar_cod = np.ones_like(self.model_time)
        
        print(self.z.shape,self.close_off_depth)
        for i in range(self.z.shape[0]):
            idx = int(np.where(self.z[i, 1:] == self.close_off_depth[i])[0])
            self.d15N_cod[i] = self.d15N[i,idx]
            #self.d40Ar_cod[i] = self.d40Ar[i,idx]
            self.T_cod[i] = self.temperature[i,idx]
    
        #print(self.d15N_cod.shape)
        self.delta_temp = self.climate[:,2] - self.T_cod
        print(self.T_cod,'Testing NOW')
        fig,ax = plt.subplots(1,tight_layout=True)
        #plt.plot(self.delta_temp)
        #plt.plot(self.climate[:,2])
        labelFont = 16
        legFont = 10
        axFont = 16
        time_labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$', '$t_4$', '$t_5$', '$t_6$','$t_f$']

        Times = [250,925,2500,3000,4500,6400]
        for i in range(len(Times)):

            ax.axvline(x=self.model_time[Times[i]],linestyle='dashed',color='lightblue')
            ax.annotate(time_labels[i], xy=(self.model_time[Times[i]]+100,68), rotation=0, verticalalignment='top')
        ax.plot(self.z[:, 0],self.close_off_depth,linewidth=0.7)
        ax22 = ax.twinx()
        
        ax22.plot(self.z[:, 0],self.delta_temp,'r',linewidth=0.7)
        ax.set_title('f)',fontsize=20)
        ax.grid(linestyle=':', color='gray', lw='0.3')
        ax.set_ylabel("Close-off-depth [m]",color='b',fontsize=labelFont)
        ax.set_xlabel("Model-time [yr]",fontsize=labelFont)
        ax.tick_params(axis='both', which='major', labelsize=axFont)
        ax.tick_params(axis='both', which='minor', labelsize=axFont-2)
        ax22.tick_params(axis='both', which='major', labelsize=axFont)
        ax22.tick_params(axis='both', which='minor', labelsize=axFont-2)
        ax.invert_yaxis()
        ax22.set_ylabel(r"Temperature grad. [K]",color='r',fontsize=labelFont)
        
        plt.show()
        plt.savefig('CoD_plots/Test.png',dpi=300)
        #"self.close_off_depth = savgol_filter(self.close_off_depth, window_length=51,polyorder=3)
        #self.d15N_cod = savgol_filter(self.d15N_cod, window_length=51, polyorder=3)
        #self.d40Ar_cod = savgol_filter(self.d40Ar_cod, window_length=51, polyorder=3)
        #self.d15N_codm = np.mean(self.d15N_cod.reshape(-1,3), axis=1)
        #self.d40Ar_codm = np.mean(self.dAr40_cod.reshape(-1,3), axis=1)
        #self.model_time[0] = 0    
        f.close()
       
        return
    
    
    def plotting(self):
        rows,cols = 3,2
        self.fig,self.ax = plt.subplots(rows,cols,figsize=(20,40), tight_layout=True)
        labelFont = 16
        legFont = 10
        axFont = 16
        
        self.ax[0,0].set_xlim(self.model_time[0],self.model_time[-1])
        #self.ax[0,0].set_ylim(230,255)
        self.ax[2,0].set_xlim(self.model_time[0],self.model_time[-1])
        self.ax[2,1].set_xlim(self.model_time[0],self.model_time[-1])
        
        #Times = [1000,5000,10000,12500,15000,16000,17000-1]
        #Times = [500, 1500, 2000, 2500, 3000, 4000, 4990]
        Times = [250,925,2500,3000,4500,6400]
        
        #Times = [100,250,400,500,600,800,1000]
        #Times = [200,500,1000,2000,2500,3000,3100]
        #Times = [10,20,50,100,200,400,492]
        cmap_interval = np.linspace(0,1,len(Times))
        time_labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$', '$t_4$', '$t_5$', '$t_6$','$t_f$']
        print(self.model_time[Times])
        
        self.ax02 = self.ax[0,0].twinx()
        self.ax[0,0].set_ylabel("Temperature[K]",color=cmap(cmap_interval[3]),fontsize=labelFont)
        self.ax02.set_ylabel("Accumulation [$\mathrm{m~y}^{-1}$]",color=cmap(cmap_interval[1]),fontsize=labelFont)
        self.ax[0,0].set_xlabel("Model-time [yr]",fontsize=labelFont)
        #self.ax[0,0].vlines(Times, 0,1 , transform=self.ax[0,0].get_xaxis_transform(), colors='b',linestyles='dashed',alpha=0.5)
        
        for i in range(len(Times)):
            self.ax[0,0].annotate(time_labels[i], xy=(self.model_time[Times[i]]+100,233), rotation=0, verticalalignment='top')
        print(self.model_time[Times])
        
        self.ax[0,1].set_ylabel("Depth [m]",fontsize=labelFont)
        self.ax[0,1].set_xlabel("Density $[\mathrm{kg m^{-3}}]$",fontsize=labelFont)
        
        self.ax[1,0].set_ylabel("Depth [m]",fontsize=labelFont)
        self.ax[1,0].set_xlabel("$\delta^{15}N$ [‰]",fontsize=labelFont)
        
        #self.ax10 = self.ax[1,0].twinx()
        
        self.ax[2,0].set_ylabel("$\delta^{15}N_{cod}$ [‰]",color=cmap(cmap_interval[2]),fontsize=labelFont)
        #self.ax10.set_ylabel("$\delta^{40}Ar_{cod}$ [‰]",color=cmap(cmap_interval[1]),fontsize=labelFont)
        self.ax[2,0].set_xlabel("Model-time [yr]",fontsize=labelFont)
        
        self.ax[1,1].set_ylabel("Depth [m]",fontsize=labelFont)
        self.ax[1,1].set_xlabel("Temperature [K]",fontsize=labelFont)
        
        self.ax[2,1].set_ylabel("Close-off-depth [m]",fontsize=labelFont)
        self.ax[2,1].set_xlabel("Model-time [yr]",fontsize=labelFont)
        
        
        
        print(self.model_time[Times])
        #print(Times)
        self.ax[0,0].grid(linestyle=':', color='gray', lw='0.3')
        self.ax[0,1].grid(linestyle=':', color='gray', lw='0.3')
        self.ax[2,0].grid(linestyle=':', color='gray', lw='0.3')
        self.ax[1,0].grid(linestyle=':', color='gray', lw='0.3')
        self.ax[1,1].grid(linestyle=':', color='gray', lw='0.3')
        self.ax[2,1].grid(linestyle=':', color='gray', lw='0.3')
        
        self.ax[0,0].set_title('a)',fontsize=20)
        self.ax[0,1].set_title('b)',fontsize=20)
        self.ax[1,0].set_title('c)',fontsize=20)
        self.ax[1,1].set_title('d)',fontsize=20)
        self.ax[2,0].set_title('e)',fontsize=20)
        self.ax[2,1].set_title('f)',fontsize=20)
        
        #print(Times)
        #print(self.d15N.shape)
        self.ax[0,0].plot(self.climate[:,0],self.climate[:,2],color=cmap(cmap_interval[3]))
        self.ax02.plot(self.climate[:,0],self.climate[:,1],color=cmap(cmap_interval[1]))
        for i in range(len(Times)):
            self.ax[0,1].plot(self.rho[Times[i]][1:], self.z[Times[i]][1:],color=cmap(cmap_interval[i]),linestyle='--',linewidth=0.7,label=time_labels[i])
            self.ax[1,0].plot(self.d15N[Times[i]][1:], self.z[Times[i]][1:],color=cmap(cmap_interval[i]),linestyle='-',linewidth=0.7,label=time_labels[i])
            self.ax[1,1].plot(self.temperature[Times[i]][1:], self.z[Times[i]][1:],color=cmap(cmap_interval[i]),linestyle='-',linewidth=0.7,label=time_labels[i])
        #print(self.d15N_cod.shape,self.close_off_depth.shape)
            self.ax[0,0].axvline(x=self.model_time[Times[i]],linestyle='dashed',color='lightblue')
            self.ax[2,0].axvline(x=self.model_time[Times[i]],linestyle='dashed',color='lightblue')
            self.ax[2,1].axvline(x=self.model_time[Times[i]],linestyle='dashed',color='lightblue')
        print(self.close_off_depth[Times])
        self.ax[2,0].plot(self.model_time,self.d15N_cod,color='r',linewidth=0.7)
        #self.ax10.plot(self.model_time,self.d40Ar_cod/4,color=cmap(cmap_interval[1]),linewidth=0.7)
        self.ax[2,1].plot(self.z[:, 0],self.close_off_depth,linewidth=0.7)
        self.ax22 = self.ax[2,1].twinx()
        
        self.ax22.plot(self.z[:, 0],self.delta_temp,'r',linewidth=0.7)
        self.ax22.set_ylabel(r"$\Delta$T [K]",color='r',fontsize=labelFont)
        #self.ax[0,0].set_ylim(230,250)
        self.ax[0,1].legend(loc='lower left',fontsize=legFont)
        #self.ax[0,2].legend(loc='lower left',fontsize=legFont)
        #self.ax[1,1].legend(loc='lower left',fontsize=legFont)
        self.ax[0,1].invert_yaxis()
        self.ax[1,0].invert_yaxis()
        self.ax[1,1].invert_yaxis()
        self.ax[2,1].invert_yaxis()
        
        
        
        
        
        
        
        
        self.ax02.tick_params(axis='both', which='major', labelsize=axFont)
        self.ax02.tick_params(axis='both', which='minor', labelsize=axFont-2)
        self.ax22.tick_params(axis='both', which='major', labelsize=axFont)
        self.ax22.tick_params(axis='both', which='minor', labelsize=axFont-2)
        for a in self.ax.flatten():
            a.tick_params(axis='both', which='major', labelsize=axFont)
            a.tick_params(axis='both', which='minor', labelsize=axFont-2)
        
        
        #self.ax[0,0].set_yticks([232,234,236,238,240,242])
        self.ax02.set_yticks([0.180,0.185,0.190,0.195,0.2])
        self.ax[0,0].set_xticks([2000,4000,6000])
        self.ax[1,0].set_xticks([0.0,0.2,0.4])
        self.ax[1,1].set_xticks([235,240])
        self.ax[2,0].set_yticks([0.0,0.1,0.2,0.3,0.4])
        #self.ax[2,1].set_yticks([70,75,80,85,90,95])

        self.ax[0,1].set_xticks([400,600,800])
        
        
        #self.fig.tight_layout(pad=2)
        
        self.ax22.yaxis.set_major_locator(plt.MaxNLocator(5))

        #self.fig.tight_layout(pad=2)
        
        
        
        
        
        
        
        #self.fig.constrained_layout(pad=2)
        
        
        
        #print(self.model_time[1],self.model_time.min())
                    






folder = './CFM/CFM_main/CFMinput/DO_event'
#Folder = np.array(['Temp_ramp'])
Folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
rfolder = 'CFM/CFM_main/CFMoutput/DO_event'
    
for j in range(1):
    print(Folder[j])
    if os.path.exists(rfolder + "/CFMresults.hdf5"):
        Current_plot = CoD_plotter(filepath=rfolder)
        Current_plot.plotting()
        plt.savefig('CoD_plots/'+str(Folder[j])+'.png',dpi=300)
        #plt.close('all')
    else:
        print("Results file does not exist")
#plt.close('all')










