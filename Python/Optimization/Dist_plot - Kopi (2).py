# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:25:28 2023

@author: jespe
"""

from matplotlib import pyplot as plt
from Kindler_fit_Clear import input_file, expfunc
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.ticker as mtick
import h5py as h5py
for fcnt in range(1,4,1):
    fname = f'file_{fcnt}.h5'
    with h5py.File(fname,'w') as h5fw:
        arr = np.random.random(10*10).reshape(10,10)
        h5fw.create_dataset('data',data=arr)
from pathlib import Path

S = 100
cost_func = np.zeros(S)
d15N = np.zeros(S)
count = np.zeros(S)
Temp = np.zeros(S)
np.random.seed(42)

#0.597, 0.463
#217.97, 229.96
#Point_N = np.array([0.53,0.597,0.375,0.297])
#Point_T = np.array([215.06,217.97,235,244.99])
#Point_A = np.array([0.0284,0.0535,0.1607,0.2621])

plt.close('all')
Models = ['Ulti1','Ulti2','Ulti3']
Dist = ['Dist' + str(i) for i in range(4)]
Dists2 = np.array([1,2,4,5])
import seaborn as sns
sns.set_theme()
palette = sns.color_palette(None,3)
from astropy.visualization import hist
a = u'u\00B1'
a.encode('utf-8')


even = np.arange(0,7,2)
odd = np.arange(1,8,2)  
Input_temp,Input_acc,Beta = input_file()

dfc = pd.read_csv('resultsFolder/Integer_diffu.csv',sep=',')
Modela = ['HLD','HLS','BAR','GOU']
title = ['Only temp','Including rho','Rho and Deff']
'''
mus = np.zeros((4,8))
mus_1 = np.zeros((4,8))
# Loop over H5 files and load into a dataframe
for i,val in enumerate(Dists2):
    #j2 = Dists2[i]
    #dfc = pd.read_csv('resultsFolder/Integer_diffu.csv',sep=',')
    
    fig, ax = plt.subplots(nrows=3,ncols=4,figsize=(15,12), constrained_layout=True)
    for j in range(len(Modela)):
        #j1 = Dists2[j]
        Csv_point = dfc[str(Modela[j])]
        Data_d15N = np.random.normal(Csv_point[val],0.02,size=100)

        #Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())][:S]
        
        print(val,Modela[j])
        #for z,file in enumerate(glob.iglob('resultsFolder/Version1/' + str(Models[j]) + '/' + str(Dist[i]) + '/*.h5')):  
        for z in range(S):
            file = 'resultsFolder/ALL_results/Ulti_models/'  + str(Modela[j]) + '/' + str(Dists2[i]) + '/Point' + str(z) + '.h5'
            #print(file)
            
            with h5py.File(file, 'r') as h5fr:
                #print(h5fr.keys())
                #print(2)
                cost_func[z] = h5fr['cost_func'][-1]
                d15N[z] = h5fr['d15N@CoD'][-1]
                count[z] = h5fr['count'][-1]
                Temp[z] = h5fr['temp'][-1]
                #print(np.mean(Temp))
       
       
        #table = pd.DataFrame(xdata).reset_index()
        #print(table)
        bins = 'freedman'
        
        hist(Data_d15N, bins=bins, ax=ax[0,j],histtype='stepfilled',color = palette[0],label=r'Input $\delta^{15}$N' if j ==3 else '')
        hist(Temp, bins=bins, ax=ax[1,j],histtype='stepfilled',color = palette[1],label='Output Temperature' if j ==3 else '')
        hist(count, bins=bins, ax=ax[2,j],histtype='stepfilled',color = palette[2],label='Iteration count' if j ==3 else '')
        
        mu = np.mean(Temp)
        std = np.std(Temp)
        ax[1,j].axvline(mu,color='g',linestyle="--",label='Mean' if j ==3 else '')
        ax[1,j].hlines(y=1,xmin=mu-std,xmax=mu+std,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')
        ax[1,j].axvline(Input_temp[val],color='k',linestyle="--",label='Expected Temperature' if j ==3 else '')
        ax[1,j].text(0, 0.9, r'$\mu$:{0:.2f}'.format(mu), size=14, ha='left', va='center',transform=ax[1,j].transAxes)
        ax[1,j].text(0, 0.8, r'$\sigma$:{0:.1f}'.format(std), size=14, ha='left', va='center',transform=ax[1,j].transAxes)
        ax[1,j].text(0, 0.7, r'$T_r$:{0:.2f}'.format(Input_temp[val]), size=14, ha='left', va='center',transform=ax[1,j].transAxes)
        
        
        
        
        mus[i,even[j]] = '{0:.3f}'.format(mu) 
        mus[i,odd[j]] = '{0:.2f}'.format(std)
        print(mu,Input_temp[val])
        mua = np.mean(Data_d15N)
        stda = np.std(Data_d15N)
        ax[0,j].axvline(mua,color='k',linestyle="--",label='Mean' if j ==3 else '')
        ax[0,j].hlines(y=1,xmin=mua-stda,xmax=mua+stda,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')

        mus_1[i,even[j]] = '{0:.3f}'.format(mua) 
        mus_1[i,odd[j]] = '{0:.2f}'.format(stda)


        
        ax[0,j].set_xlabel(u'$\delta^{15}$N [\u2030]',fontsize=20)
        ax[1,j].set_xlabel('Temperature [K]',fontsize=20)
        #ax[2,j].set_xlabel('Point nr.',fontsize=18)
        #ax[3,j].set_xlabel('Point nr.')
        ax[2,j].set_xlabel('Iteration count',fontsize=20)
        #ax[3,j].set_ylabel('Cost function')
        ax[0,j].set_title(str(Modela[j]),fontsize=20)

        for l, axes in enumerate(fig.axes):
            axes.tick_params(axis='both', which='major', labelsize=18)
        #t.set_x(1.0)
        #ax[3,j].yaxis._update_offset_text_position = types.MethodType(bottom_offset, ax.xaxis)
        if j == 3:
            for z in range(3):
                box = ax[z,j].get_position()
                #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, bo6x.height])

    # Put a legend to the right of the current axis
                ax[z,j].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
    
    
    
    
    newpath = 'Steady_state/Densification_schemes' 
    path = Path(newpath)
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(newpath+ '/'+str(val)+'.png',dpi=300)
        


Models = ['HLD','HLS','Bar','GOU']
Output = ['Mean',r'$\sigma$']

Iter1 = [Models,Output]
Mode = ['Temp','Nitrogen']
Dist = ['Dist' + str(x) for x in np.array([1,2,4,5])]
Iter2 = [Mode,Dist]




idx = pd.MultiIndex.from_product(Iter2)



cols = pd.MultiIndex.from_product(Iter1)

Matrix = np.vstack((mus,mus_1))
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




#df.style.format(decimal='.', thousands=',', precision=1)
df = df.astype(str)




with open('Steady_state/Test.tex', 'w') as tf:
     tf.write(df.style.to_latex(column_format="cccccccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Equilibrium times for amplitude",
                multirow_align="c", multicol_align="c")  
              )

'''





cost_func = np.zeros((len(Dists2),len(Modela),S))
d15N = np.zeros((len(Dists2),len(Modela),S))
count = np.zeros((len(Dists2),len(Modela),S))
Temp = np.zeros((len(Dists2),len(Modela),S))
Data_d15N = np.zeros((len(Dists2),len(Modela),S))

for i,val in enumerate(Dists2):
    for j in range(len(Modela)):


        for z in range(S):
            file = 'resultsFolder/ALL_results/Ulti_models/'  + str(Modela[j]) + '/' + str(Dists2[i]) + '/Point' + str(z) + '.h5'
    #print(file)
    
            with h5py.File(file, 'r') as h5fr:
        #print(h5fr.keys())
        #print(2)
                cost_func[i,j,z] = h5fr['cost_func'][-1]
                d15N[i,j,z] = h5fr['d15N@CoD'][-1]
                count[i,j,z] = h5fr['count'][-1]
                Temp[i,j,z] = h5fr['temp'][-1]



    


def axg_lines(j):
    mu = np.mean(Temp[-1,j,:])
    std = np.std(Temp[-1,j,:])
    ax.axvline(mu,color='g',linestyle="--",label='Mean' if j ==3 else '')
    ax.hlines(y=1,xmin=mu-std,xmax=mu+std,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')
    ax.axvline(Input_temp[5],color='k',linestyle="--",label='Expected Temperature' if j ==3 else '')
    ax.text(0, 0.9, r'$\mu$:{0:.2f}'.format(mu), size=14, ha='left', va='center',transform=ax.transAxes)
    ax.text(0, 0.8, r'$\sigma$:{0:.1f}'.format(std), size=14, ha='left', va='center',transform=ax.transAxes)
    ax.text(0, 0.7, r'$T_r$:{0:.2f}'.format(Input_temp[5]), size=14, ha='left', va='center',transform=ax.transAxes)
    ax.set_xlabel('Temperature [K]',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
def ax_d15N_line(j):
    mua = np.mean(Data_d15N)
    stda = np.std(Data_d15N)
    ax.axvline(mua,color='k',linestyle="--",label='Mean' if j ==3 else '')
    ax.hlines(y=1,xmin=mua-stda,xmax=mua+stda,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')
    ax.set_xlabel(u'$\delta^{15}$N [\u2030]',fontsize=20)
    ax.set_title(str(Modela[j]),fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    
    
def ax_count(j):
    ax.set_xlabel('Iteration count',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)







bins = 'freedman'

from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
print(Temp.min())
fig = plt.figure(figsize=(15,15))
sps = GridSpec(3,4,figure=fig)

ax = fig.add_subplot(sps[0,0])
Csv_point = dfc[str(Modela[0])]
Data_d15N = np.random.normal(Csv_point[5],0.02,size=100)
hist(Data_d15N, bins=bins, ax=ax,histtype='stepfilled',color = palette[0])
ax_d15N_line(0)


Csv_point = dfc[str(Modela[1])]
Data_d15N = np.random.normal(Csv_point[5],0.02,size=100)
ax = fig.add_subplot(sps[0,1])
hist(Data_d15N, bins=bins, ax=ax,histtype='stepfilled',color = palette[0])
ax_d15N_line(1)




Csv_point = dfc[str(Modela[2])]
Data_d15N = np.random.normal(Csv_point[5],0.02,size=100)
ax = fig.add_subplot(sps[0,2])
hist(Data_d15N, bins=bins, ax=ax,histtype='stepfilled',color = palette[0])
ax_d15N_line(2)




Csv_point = dfc[str(Modela[3])]
Data_d15N = np.random.normal(Csv_point[5],0.02,size=100)
ax = fig.add_subplot(sps[0,3])
hist(Data_d15N, bins=bins, ax=ax,histtype='stepfilled',color = palette[0],label=r'Input $\delta^{15}$N')
ax_d15N_line(3)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)



ax = fig.add_subplot(sps[1,0])
hist(Temp[-1,0,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[1])
axg_lines(0)

ax = fig.add_subplot(sps[1,1])
hist(Temp[-1,1,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[1])
axg_lines(1)

ax = fig.add_subplot(sps[1,2])
hist(Temp[-1,2,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[1])
axg_lines(2)

ax = brokenaxes(d=0.000001,xlims=((0,1), (235,245.3)),subplot_spec=sps[1,3])
ax.hist(Temp[-1,3,:], bins = [0]+list(np.linspace(235,246,8+1)),label='Output Temperature',histtype='stepfilled',color=palette[1])
mu = np.mean(Temp[-1,j,:])
std = np.std(Temp[-1,j,:])
ax.axvline(mu,color='g',linestyle="--",label='Mean' if j ==3 else '')
ax.hlines(y=1,xmin=mu-std,xmax=mu+std,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')
ax.axvline(Input_temp[5],color='k',linestyle="--",label='Expected Temperature' if j ==3 else '')
ax.text(235.1,27 , r'$\mu$:{0:.2f}'.format(mu), size=14, ha='left', va='center')
ax.text(235.1, 24, r'$\sigma$:{0:.1f}'.format(std), size=14, ha='left', va='center')
ax.text(235.1, 21, r'$T_r$:{0:.2f}'.format(Input_temp[5]), size=14, ha='left', va='center')
ax.set_xlabel('Temperature [K]',fontsize=20,labelpad=30)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)

ax = fig.add_subplot(sps[2,0])
hist(count[-1,0,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[2])
ax_count(0)

ax = fig.add_subplot(sps[2,1])
hist(count[-1,1,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[2])
ax_count(1)


ax = fig.add_subplot(sps[2,2])
hist(count[-1,2,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[2])
ax_count(2)

#x = np.random.poisson(3, 1000)

ax = fig.add_subplot(sps[2,3])
hist(count[-1,3,:], bins=bins, ax=ax,histtype='stepfilled',color = palette[2],label='Iteration count')
ax_count(3)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)

fig.tight_layout()
    #for l, axes in enumerate(fig.axes):
    #if j == 3:
    #    for z in range(3):
    #        box = ax[z,j].get_position()
            #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, bo6x.height])

# Put a legend to the right of the current axis
            

newpath = 'Steady_state/Densification_schemes/' 
path = Path(newpath)
path.mkdir(parents=True, exist_ok=True)
fig.savefig(newpath+ 'Fixed.png',dpi=300)
    





    
    
    
    
    

 

print(mu,Input_temp[5])
    



    
    


