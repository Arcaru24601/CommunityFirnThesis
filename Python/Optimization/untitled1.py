# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 01:30:06 2023

@author: jespe
"""


from matplotlib import pyplot as plt
from Kindler_fit_Clear import input_file, expfunc
import numpy as np
import pandas as pd
import glob
import matplotlib.ticker as mtick
import h5py as h5py
for fcnt in range(1,4,1):
    fname = f'file_{fcnt}.h5'
    with h5py.File(fname,'w') as h5fw:
        arr = np.random.random(10*10).reshape(10,10)
        h5fw.create_dataset('data',data=arr)

S = 200
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
Dists2 = np.array([5])
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




def get_data(param,S):
    for z in range(S):
        file = 'resultsFolder/' + str(param) + '/HLD/' + str(Dists2[i]) + '/Point' + str(z) + '.h5'
        
        with h5py.File(file, 'r') as h5fr:
            #print(h5fr.keys())
            #print(2)
            cost_func[z] = h5fr['cost_func'][-1]
            d15N[z] = h5fr['d15N@CoD'][-1]
            count[z] = h5fr['count'][-1]
            Temp[z] = h5fr['temp'][-1]
            #print(np.mean(Temp))
    return cost_func, d15N, count, Temp





mus = np.zeros((13,8))
mus_1 = np.zeros((13,8))
# Loop over H5 files and load into a dataframe
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,7), constrained_layout=True)
for i,val in enumerate(Dists2):
    #j2 = Dists2[i]
    Csv_point = dfc['HLD']
    Data_d15N = np.random.normal(Csv_point[val],0.02,size=200)

    
    for j in range(len(Models)):
        #j1 = Dists2[j]
       
        #Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())][:S]

        print(val,Models[j])
        #for z,file in enumerate(glob.iglob('resultsFolder/Version1/' + str(Models[j]) + '/' + str(Dist[i]) + '/*.h5')):  
        cost_func,d15N,count, Temp = get_data(Models[j], S)
       
        #table = pd.DataFrame(xdata).reset_index()
        #print(table)
        bins = 'scott'
        hist(Temp, bins=bins, ax=ax[j],histtype='stepfilled',color = palette[j],alpha=1,label=str(Dists2[i]) if j ==3 else '')
        #hist(count, bins=bins, ax=ax[2,j],histtype='stepfilled',color = palette[2],label='Iteration count' if j ==3 else '')
        
        mu = np.mean(Temp)
        std = np.std(Temp)
        ax[j].axvline(mu,color='g',linestyle="--",label='Mean' if j ==3 else '')
        ax[j].hlines(y=1,xmin=mu-std,xmax=mu+std,color='y',linestyle="-",label=r'1$\sigma$ deviation' if j ==3 else '')
        ax[j].axvline(Input_temp[val],color='k',linestyle="--",label='Expected Temperature' if j ==3 else '')
        ax[j].text(0, 0.9, r'$\mu$:{0:.2f}'.format(mu), size=14, ha='left', va='center',transform=ax[j].transAxes)
        ax[j].text(0, 0.8, r'$\sigma$:{0:.1f}'.format(std), size=14, ha='left', va='center',transform=ax[j].transAxes)
        ax[j].text(0, 0.7, r'$T_r$:{0:.2f}'.format(Input_temp[val]), size=14, ha='left', va='center',transform=ax[j].transAxes)
        
        if j == 1:
            cost_func1,d15N1,count1, Temp1 = get_data(Models[0], S)
            hist(Temp1, bins=bins, ax=ax[j],histtype='stepfilled',color = palette[0],alpha=0.5,label=str(Dists2[0]) if j ==3 else '')

        
        if j == 2:
            cost_func1,d15N1,count1, Temp1 = get_data(Models[0], S)
            hist(Temp1, bins=bins, ax=ax[j],histtype='stepfilled',color = palette[0],alpha=0.5,label=str(Dists2[0]) if j ==3 else '')

            cost_func2,d15N2,count2, Temp2 = get_data(Models[1], S)
            hist(Temp2, bins=bins, ax=ax[j],histtype='stepfilled',color = palette[1],alpha=0.5,label=str(Dists2[1]) if j ==3 else '')

        
        
        
        
        
        
        
        
        
        ax[j].set_xlabel('Temperature [K]',fontsize=20)
        ax[j].set_xlabel('Temperature [K]',fontsize=20)

        for l, axes in enumerate(fig.axes):
            axes.tick_params(axis='both', which='major', labelsize=18)
        #t.set_x(1.0)
        #ax[3,j].yaxis._update_offset_text_position = types.MethodType(bottom_offset, ax.xaxis)
        if j == 3:
            for z in range(2):
                box = ax[j].get_position()
                #ax[i,j].set_position([box.x0, box.y0, box.width * 0.8, bo6x.height])

    # Put a legend to the right of the current axis
                ax[j].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
                #ax[0,3].legend(loc='upper right')
                #ax[1,3].legend(loc='upper right')
                #ax[2,3].legend(loc='upper right')
    #fig.legend(ncol=3,loc='lower center')
    plt.savefig('Plots_ulti2/Dist'+str(val)+'fix.png',dpi=300)
        


Models = ['HLD','HLS','Bar','GOU']
Output = ['Mean',r'$\sigma$']

Iter1 = [Models,Output]
Mode = ['Temp','Nitrogen']
Dist = ['Dist' + str(x) for x in np.arange(0,25,2)]
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




with open('Stuff.tex', 'w') as tf:
     tf.write(df.style.to_latex(column_format="cccccccccc", position="h", position_float="centering",
                hrules=True, label="table:5", caption="Equilibrium times for amplitude",
                multirow_align="c", multicol_align="c")  
              )




    
