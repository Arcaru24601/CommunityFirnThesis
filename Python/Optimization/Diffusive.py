# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:14:52 2023

@author: Jesper Holm
"""


import h5py as h5
import os
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.cm.get_cmap('Dark2')
import seaborn as sns 
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#import reader as re
sns.set()


cmap_intervals = np.linspace(0, 1, 6)
from pathlib import Path
plt.close('all')

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
    
class CoD_plotter():

    def __init__(self,j,filepath=None,sub=None,KtC=False):
        self.filepath = filepath
        self.KtC = KtC
        self.sub = sub
        
    def CoD_out(self):
        fig,ax = plt.subplots(constrained_layout=True)
        label = [x[:-1] for x in Diffus]
        for i in range(len(self.filepath)):
            if not os.path.exists(self.filepath[i]):
                print('Results file does not exist', self.filepath[i][40:])
                continue
            f = h5.File(self.filepath[i])
            #print(self.filepath)
            #print(self.filepath)
            z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            self.temperature = f['temperature'][:]
            if self.close_off_depth[i] < 0:
                continue
            #print(f.keys())
            diffu = f['diffusivity'][:]
            #plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
            ax.plot(diffu[-1,1:],z[-1,1:],label=label[i],color=cmap(cmap_intervals[i]))
        ax.tick_params(axis='both', which='major', labelsize=16)

            #ax2.set_ylim(ax1.get_ylim())
                #ax2 = ax1.twiny()    
                #plt.ylim((0.2,0.6))
        ax.set_xlabel(r'Effective diffusivity [m$^2$ s$^{-1}$]',fontsize=18)
        ax.set_ylim(-1,120)
        ax.set_xlim(right=1e-5)

            #.set_xlabel(r'Accumulation rate [m yr$^{-1}$]',fontsize=14)


            #box = ax1.get_position()
                        #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
        ax.legend(loc='best',fontsize=14)
        ax.set_ylabel('Depth [m]',fontsize=18)
        ax.invert_yaxis()
            
            
        #print(self.close_off_depth[-1])
        #plt.show()
        plt.savefig('Optimization/Noise/Diffs/' +str(self.sub[:-1]) + '.png' ,dpi=300)
        f.close()
        
            
            
            
            
            
        return z,diffu
            
            
              
        
         
        
        
    
Models = ['HLdynamic/','HLSigfus/','Barnola1991/','Goujon2003/']  
Diffus = ['Freitag/', 'Schwander/', 'Severinghaus/', 'Witrant/', 'Battle/', 'Adolph/']
  
def folder_gen(Fold,FileFlag):
    X = [Fold]
    X2 = Diffus
    
    if FileFlag == True:
        X = [x[:-1] for x in X]
        X2 = [x[:-1] for x in X2]
        Folder = [(i2+i) for i in X for i2 in X2]
    else:
        #X2 = [s + '/' for s in X2]
        Folder = [(i2+i) for i in X for i2 in X2]
    
    return Folder 
 
os.chdir('../')
folder = 'CFM/CFM_main/CFMinput/Noise/Round6/'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

sub2 = [x[:-1] for x in sub_folders]
sub_folders = [x + '/' for x in sub_folders]



rfolder = 'CFM/CFM_main/CFMoutput/Noise/bco/'

for j in range(len(sub_folders)):
#for j in range(1):
    T = folder_gen(sub_folders[j],False)
    P = folder_gen(sub_folders[j],True)
    path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
    #print(path)
    print(sub_folders[j])
    Current_plot = CoD_plotter(j,path,sub_folders[j])
    z,diffu = Current_plot.CoD_out()
    plt.close('all')




