# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:37:17 2023

@author: Jesper Holm
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.ticker as mtick
import h5py as h5py
from matplotlib import pyplot as plt
from Kindler_fit_ODR import input_file,expfunc
K = 25
from scipy.stats import t
cost_func = np.zeros(K)
d15N = np.zeros(K)
count = np.zeros(K)
Temp = np.zeros(K)
np.random.seed(42)

#0.597, 0.463
#217.97, 229.96
Point_N = np.array([0.53,0.597,0.375,0.297])
Point_T = np.array([215.06,217.97,235,244.99])
#Point_A = np.array([0.0284,0.0535,0.1607,0.2621])

plt.close('all')
Models = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']
#Dist = ['Dist' + str(i) for i in range(4)]

import seaborn as sns
sns.set_theme()
palette = sns.color_palette(None,3)
from astropy.visualization import hist
a = u'u\00B1'
a.encode('utf-8')
Modela = ['HLD','HLS','BAR','GOU']

Input_temp,Input_acc,Beta = input_file(num=25)

even = np.arange(0,7,2)
odd = np.arange(1,8,2)  
df = pd.read_csv('resultsFolder/out_model.csv',sep=',')
#df = pd.read_csv('resultsFolder/out_diffu.csv',sep=',')
mus = np.zeros((4,8))
mus_1 = np.zeros((4,8))


# Loop over H5 files and load into a dataframe
for i in range(len(Models)):
    
    #s = np.random.normal(Point_N[i],0.02,size=2000)
    #Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())][:250]
    Data_d15N = df[Modela[i]]
    fig, ax = plt.subplots(nrows=3,figsize=(10,7),constrained_layout=True)
    print('resultsFolder/Version5/' + str(Models[i]) + '/*.h5')
    for z in range(len(Input_temp)):
        file = 'resultsFolder/Version5/' + str(Models[i]) + '/Point' + str(z) + '.h5'
        #print(file)
        
        try:
            with h5py.File(file, 'r') as h5fr:
                #print(h5fr.keys())
                #print(2)
                cost_func[z] = h5fr['cost_func'][-1]
                d15N[z] = h5fr['d15N@CoD'][-1]
                count[z] = h5fr['count'][-1]
                Temp[z] = h5fr['temp'][-1]
        except Exception as e: print(e)
    bins = 'freedman'
    #if i == 3:
    #    Temp[np.where(Temp==0)] = float('NaN')
    #hist(Data_d15N, bins=bins, ax=ax[0],histtype='stepfilled',color = palette[0],label=r'Input $\delta^{15}$N')
    #hist(Temp, bins=bins, ax=ax[1],histtype='stepfilled',color = palette[1],label='Output Temperature')
    #hist(count, bins=bins, ax=ax[2],histtype='stepfilled',color = palette[2],label='Iteration count')
    #ax[0]
    x = Temp - Input_temp
    m = np.nanmean(x)
    s = np.nanstd(x)
    ax[0].plot(Input_temp,Data_d15N-d15N,'o',label='Deviation in d15N')
    
    ax[1].plot(Input_temp,Temp-Input_temp,'o',label='Deviation in temperature')
    ax[2].plot(Input_temp,count,'o',label='Iteration count')
    
    #dof = len(x)-1
    #conf = 0.95
    #t_crit = np.abs(t.ppf((1-conf)/2,dof))
    #ci = s*t_crit/np.sqrt(len(x))
    ax[1].fill_between(Input_temp, (m-s), (m+s),alpha=0.25,label='std ' + f'{s:.3}')
    ax[1].plot(Input_temp,np.full_like(Input_temp,m),label='Mean '+ f'{m:.3}')
    #ax.plot(Input_temp,'o')
    #mu = np.mean(Temp)
    #std = np.std(Temp,ddof=1)
    #ax[1].axvline(mu,color='k',linestyle="--",label='Mean')
    #ax[1].hlines(y=1,xmin=mu-std,xmax=mu+std,color='y',linestyle="-",label=r'1$\sigma$ deviation')
    #ax[1].axvline(Point_T[i],color='g',linestyle="--",label='Expected Temperature')
     
    #mus[i,even[j]] = '{0:.2f}'.format(mu) 
    #mus[i,odd[j]] = '{0:.1f}'.format(std)
    #print(mu)
    #mua = np.mean(Data_d15N)
    #stda = np.std(Data_d15N,ddof=1)
    #ax[0].axvline(mua,color='k',linestyle="--",label='Mean')
    #ax[0].hlines(y=1,xmin=mua-stda,xmax=mua+stda,color='y',linestyle="-",label=r'1$\sigma$ deviation')
     #ax[1,j].axvline(Point_T[i],color='g',linestyle="--")
     
    #mus_1[i,even[j]] = '{0:.3f}'.format(mua) 
    #mus_1[i,odd[j]] = '{0:.2f}'.format(stda)

     
     
     #hist(cost_func, bins=bins, ax=ax[3,j], histtype='stepfilled',color=palette[3])
     #ax[0,j].hist(Data_d15N,bins='fd',color = palette[0]) ### d15N input
     #ax[1,j].hist(Temp,bins='fd',color = palette[1]) ### Temperature output
     #ax[2,j].hist(count,bins='fd',color = palette[2]) ### Accumulation
     #ax[3,j].plot(cost_func,color=palette[3]) ### Cost function
     
    ax[0].set_ylabel(u'$\delta^{15}$N [\u2030]',fontsize=14)
    ax[1].set_xlabel('Temperature [K]',fontsize=14)
    ax[2].set_xlabel('Point nr.',fontsize=14)
    #ax[3,j].set_xlabel('Point nr.')
    ax[2].set_ylabel('Iteration count',fontsize=14)
    #ax[3,j].set_ylabel('Cost function')
    #ax[0].set_title(str(Models[i]))
     #ax[3,j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
     #ax[3,j].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
     #ax[3,j].xaxis.set_major_locator(plt.MaxNLocator(3))
     #t = ax[3,j].yaxis.get_offset_text()
     #t.set_x(1.0)
     #ax[3,j].yaxis._update_offset_text_position = types.MethodType(bottom_offset, ax.xaxis)
    
    for z in range(3):
        box = ax[z].get_position()
        ax[z].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.savefig('Plots3/Dist'+str(i)+'.png',dpi=300)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                