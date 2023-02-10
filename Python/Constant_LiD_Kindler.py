# -*- LiDing: utf-8 -*-
"""
Created on Thu Feb  9 14:02:43 2023

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
cmap_intervals = np.linspace(0, 1, 14)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
class LiD_plotter():

    def __init__(self,j,filepath=None,Models=None,KtC=False):
        
        
        self.fig, self.ax = plt.subplots(1, sharex=True, sharey=False)
        #fig.set_figheight(15)
        #fig.set_figwidth(8)
        #print(filepath)
        self.filepath = filepath
        self.KtC = KtC
        self.LiD = np.array([67,63,71,98,115])
        return
    def LiD_out(self):
        
        label = Models
        for i in range(len(self.filepath)):
            if not os.path.exists(self.filepath[i]):
                print('Results file does not exist', self.filepath[i][40:])
                continue
            f = h5.File(self.filepath[i])

            self.z = f['depth'][:]
            self.climate = f["Modelclimate"][:]
            self.model_time = np.array(([a[0] for a in self.z[:]]))
            self.lock_in_depth = f["BCO"][:, 6]
            if self.lock_in_depth[i] < 0:
                continue
        
            LiD_out[j,i] = self.lock_in_depth[i]
            LiD_diff[j,i] =  100 * (LiD_out[j,i] - self.LiD[j])/ self.LiD[j]
            self.ax.axhline(self.LiD[j],color='r',linestyle='-')

            self.ax.plot(self.model_time,self.lock_in_depth, color=cmap(cmap_intervals[i]), label=label[i])
        #self.ax.invert_yaxis()
        self.ax.grid(linestyle='--', color='gray', lw='0.5')
        self.ax.set_ylabel(r'\centering Lock-in\newline\centering depth [m]')
        self.ax.set_yticks(np.arange(30,120,step=30))
        
        self.ax.legend(loc='lower left', fontsize=8)
           
        plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        f.close()
        return LiD_out[j,:],LiD_diff[j,:]
import json

file = open('CFM/CFM_main/Constant_Forcing.json')
data = json.load(file)
Models = data['physRho_options']
Models.remove('Li2015')
rfolder = 'CFM/CFM_main/CFMoutput/Constant_Forcing_Kindler/'
x = ['NGRIP', 'NEEM', 'GRIP','DomeC','South']
x2 = Models
#print(x,x2)


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
Sites = ['NGRIP/', 'NEEM/', 'GRIP/','DomeC/','South/']
Sit = ['NGRIP', 'NEEM', 'GRIP','DomeC','South']


LiD_out = np.zeros((5,13))
LiD_diff = np.zeros((5,13))
for j in range(len(Sites)):
    T = folder_gen(Sites[j],False)
    P = folder_gen(Sites[j],True)
    path = [rfolder + m+n + '.hdf5' for m,n in zip(T,P)]
    print(Sites[j])
    Current_plot = LiD_plotter(j,path,Models)
    LiD_out2 = Current_plot.LiD_out()
    LiD_out[j,:] = LiD_out2[0]#Current_plot.plotting()
    LiD_diff[j,:] = LiD_out2[1]#Current_plot.plotting()
    
    plt.savefig('Constant_Kindler/'+str(Sit[j])+'.png',dpi=300)
    plt.close('all')  
        
Temp = np.array([241.45,244.15,241.45,218.65,223.75]).reshape((5,1))
Acc = np.array([0.19,0.216,0.23,0.027,0.076]).reshape((5,1))
LiD = np.array([67,63,71,98,115]).reshape((5,1))
Site = ['NGRIP','NEEM','GRIP','DomeC','South']
header = ['Temp','Acc','LiD', 'HLD', 'HLS','Li4','Li1','HEL','ArtS','ArtT','GOU','BAR','MOR','KM','CRO','LIG'] 
Matrix = np.concatenate((Temp, Acc, LiD, np.round(LiD_out,1)), axis=1)
Matrix2 = np.concatenate((Temp, Acc, LiD, np.round(LiD_diff,1)), axis=1)
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

with open('LiDKindler.tex', 'w') as tf:
     tf.write(df.style.to_latex(column_format="cccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Lock-in-depth for Kindler sites",
                multirow_align="t", multicol_align="r")  
              )


df2 = pd.DataFrame(data = Matrix2.T, 
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




df2.style.format(decimal='.', thousands=',', precision=1)
df2 = df2.astype(str)
#df.style.applymap(color_negative_red, subset=['NGRIP','Fuji','Siple','DE08'])


#df = df.replace(to_replace = "\.0+$",value = "", regex = True)

with open('LiDdiffKindler.tex', 'w') as tf:
     tf.write(df2.style.to_latex(column_format="cccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Lock-in-depth difference for Kindler sites",
                multirow_align="t", multicol_align="r")  
              )








