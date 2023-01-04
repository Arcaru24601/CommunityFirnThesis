# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:59:19 2022

@author: jespe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:20:44 2022

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
import reader as re
sns.set()
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, 14)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
class CoD_plotter():

    def __init__(self,j,filepath=None,Models=None,KtC=False):
        
        
        self.fig, self.ax = plt.subplots(1, sharex=True, sharey=False)
        #fig.set_figheight(15)
        #fig.set_figwidth(8)
        #print(filepath)
        self.filepath = filepath
        self.KtC = KtC
        self.CoD = np.array([78,104,57,85,74,78])
        return
    def CoD_out(self):
        
        label = Models
        for i in range(len(self.filepath)):
            if not os.path.exists(self.filepath[i]):
                print('Results file does not exist', self.filepath[i][40:])
                continue
            f = h5.File(self.filepath[i])

            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.close_off_depth = f["BCO"][:, 2]
            if self.close_off_depth[i] < 0:
                continue
        
            CoD_out[j,i] = self.close_off_depth[i]
            CoD_diff[j,i] = self.CoD[j] - CoD_out[j,i]
            self.ax.axhline(self.CoD[j],color='r',linestyle='-')

            self.ax.plot(self.model_time,self.close_off_depth, color=cmap(cmap_intervals[i]), label=label[i])
        #self.ax.invert_yaxis()
        self.ax.grid(linestyle='--', color='gray', lw='0.5')
        self.ax.set_ylabel(r'\centering Close-off \newline\centering depth [m]')
        self.ax.set_yticks(np.arange(30,120,step=30))
        
        self.ax.legend(loc='lower left', fontsize=8)
           
        plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        f.close()
        return CoD_out[j,:],CoD_diff[j,:]
import json

file = open('CFM/CFM_main/Constant_Forcing.json')
data = json.load(file)
Models = data['physRho_options']

rfolder = 'CFM/CFM_main/CFMoutput/Constant_Forcing/'
x = ['NGRIP', 'Fuji', 'Siple','DE08','DML','Test']
x2 = Models
Folder = [(i+i2) for i in x for i2 in x2]


def folder_gen(Fold,FileFlag):
    X = [Fold]
    X2 = Models
    
    if FileFlag == True:
        X = [x[:-1] for x in X]
        #X2 = [x[:-1] for x in X2]
        Folder = [(i2+i) for i in X for i2 in X2]
    else:
        X2 = [s + '/' for s in X2]
        Folder = [(i+i2) for i in X for i2 in X2]
    
    return Folder 

T_Path = folder_gen('NGRIP/',True)
T_Fold = folder_gen('NGRIP/',False)
Sites = ['NGRIP/', 'Fuji/', 'Siple/','DE08/','DML/','Test/']
Sit = ['NGRIP', 'Fuji', 'Siple','DE08','DML','Test']


CoD_out = np.zeros((6,14))
CoD_diff = np.zeros((6,14))
for j in range(len(Sites)):
    T = folder_gen(Sites[j],False)
    P = folder_gen(Sites[j],True)
    path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
    print(Sites[j])
    Current_plot = CoD_plotter(j,path,Models)
    CoD_out[j,:] = Current_plot.CoD_out()[0]#Current_plot.plotting()
    CoD_diff[j,:] = Current_plot.CoD_out()[1]#Current_plot.plotting()
    
    plt.savefig('Constant/'+str(Sit[j])+'.png',dpi=300)
    plt.close('all')  
        
Temp = np.array([242.05,215.85,247.75,254.2,234.15,215.85]).reshape((6,1))
Acc = np.array([0.19,0.028,0.13,1.2,7.0,0.19]).reshape((6,1))
CoD = np.array([78,104,57,85,74,78]).reshape((6,1))
Site = ['NGRIP','Fuji','Siple','DE08','DML','Test']
header = ['Temp','Acc','CoD', 'HLD', 'HLS','Li4','Li1','HEL','ArtS','ArtT','Li5','GOU','BAR','MOR','KM','CRO','LIG'] 
Matrix = np.concatenate((Temp, Acc, CoD, np.round(CoD_out,1)), axis=1)
Matrix2 = np.concatenate((Temp, Acc, CoD, np.round(CoD_diff,1)), axis=1)
import pandas as pd


df = pd.DataFrame(data = Matrix.T, 
                  index = header, 
                  columns = Site)


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
df = df.astype(str)
#df.style.applymap(color_negative_red, subset=['NGRIP','Fuji','Siple','DE08'])


#df = df.replace(to_replace = "\.0+$",value = "", regex = True)

with open('mytable.tex', 'w') as tf:
     tf.write(df.style.to_latex(column_format="cccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Styled LaTeX Table",
                multirow_align="t", multicol_align="r")  
              )


df = pd.DataFrame(data = Matrix2.T, 
                  index = header, 
                  columns = Site)

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
df = df.astype(str)
#df.style.applymap(color_negative_red, subset=['NGRIP','Fuji','Siple','DE08'])


#df = df.replace(to_replace = "\.0+$",value = "", regex = True)

with open('mytable2.tex', 'w') as tf:
     tf.write(df.style.to_latex(column_format="cccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Styled LaTeX Table",
                multirow_align="t", multicol_align="r")  
              )











      
