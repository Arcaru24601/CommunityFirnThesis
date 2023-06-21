# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:55:48 2023

@author: Jesper Holm
"""

import matplotlib.pyplot as plt
import numpy as np
from read_temp_acc import *
from read_d15n import *


plt.close('all')
t = np.arange(0, 10, 0.01)

y1 = 2*np.pi*t
y2 = 4*np.pi*np.sqrt(t)
data_path = 'data/NGRIP/interpolated.xlsx'





def slice_time_series(ts, start, end):
    """
    Slice the time series ts from start to end and return two arrays:
    - elements within the slice
    - elements outside the slice
    """
    slice_data = ts[start:end]
    outside_data = np.concatenate([ts[:start], ts[end:]])
    return slice_data, outside_data



#ax.legend(handles=[p1, p2, p3])




def get_d15N_data_intervals(path_data):
    df = pd.read_excel(path_data)
    data = np.flipud(np.array(df[df.columns[9]]))
    
    return data

def get_d15Na(path_data):
    
    df = pd.read_excel(path_data, sheet_name='Sheet6')
    d15n = np.flipud(np.array(df[df.columns[3]]))
    publish = np.flipud(np.array(df[df.columns[1]]))
    depth = np.flipud(np.array(df[df.columns[2]]))
    #d15n_err = np.flipud(np.array(df[df.columns[4]]))
    return d15n,publish,depth



def Make_fig():
    fig, ax = plt.subplots(figsize=(20,25),nrows=3,sharex=True)




    
    #twin1 = ax[0].twinx()
    twin2 = ax[0].twinx()
    #ax[0].xaxis.grid(True, which='minor')
    #ax[2].xaxis.grid(True, which='minor')
    #ax[1].xaxis.grid(True, which='minor')

    ax[1].yaxis.label.set_color('k')
    ax[0].yaxis.label.set_color('g')
    ax[2].yaxis.label.set_color('k')
    twin2.yaxis.label.set_color('y')

    ax[1].tick_params(axis='y', colors='k')
    ax[0].tick_params(axis='y', colors='g')
    ax[2].tick_params(axis='y', colors='k')
    twin2.tick_params(axis='y', colors='y')

  
    
    ax[0].xaxis.set_visible(False)
    #ax[2]xaxis.set_visible(False)
    ax[0].tick_params(axis='x', which='both', bottom=False)


    ax[0].spines.top.set_visible(True)
    ax[0].spines.bottom.set_visible(False)
    ax[0].spines.right.set_visible(True)

    ax[1].spines.bottom.set_visible(False)
    
    ax[1].spines.top.set_visible(False)
    ax[1].tick_params(axis='x', which='both', bottom=False)
    ax[1].spines.left.set_visible(True)
    
    
    twin2.yaxis.set_label_position("right")
    twin2.yaxis.tick_right()
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    #ax[0].spines['left'].set_visible(False)
    
    
    

    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    
    ax[2].spines.top.set_visible(False)
    twin2.spines.top.set_visible(False)
    ax[2].spines.right.set_visible(True)
    twin2.spines.bottom.set_visible(False)
    

    ax[2].spines.right.set_visible(False)

    

    
    return fig,ax,twin2


def Make_fig2():
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
    #plt.title(stock)
    #plt.ylabel('H-L')
    ax2 = plt.subplot2grid((3,1), (1,0), rowspan=1, colspan=1, sharex=ax1)
    #plt.ylabel('Price')
    #ax1  = plt.subplot2grid((2,2),(0), colspan = 1)
    #ax2  = plt.subplot2grid((2,2),(0), colspan = 1)
    #ax3  = plt.subplot2grid((2,2),(0), colspan = 1)
    
    ax2v = ax1.twinx()
    
    ax3 = plt.subplot2grid((3,1), (2,0), rowspan=1, colspan=1, sharex=ax1)
    
    ax1.yaxis.label.set_color('k')
    ax2.yaxis.label.set_color('g')
    ax2v.yaxis.label.set_color('k')
    ax3.yaxis.label.set_color('y')
    ax3.tick_params(axis='x', which='both', bottom=False)
    
    
    
    ax1.tick_params(axis='y', colors='k')
    ax2.tick_params(axis='y', colors='g')
    ax2v.tick_params(axis='y', colors='k')
    ax3.tick_params(axis='y', colors='y')
    return fig,ax1,ax2,ax2v,ax3

Start = np.array([-122296,-85501,-47501])
End = np.array([-85500,-47500,-9961.19])
color = ['blue','darkcyan','deepskyblue','lightblue']
depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)

for j in range(len(Start)):
    start_ind, end_ind = find_start_end_ind(ice_age_full, Start[j], End[j])
    print(ice_age_full[start_ind],ice_age_full[end_ind],)




d15N,g,depth = get_d15Na('data/NGRIP/supplement.xlsx')
ice_age = np.linspace(Start[0],End[-1],len(d15N))




#depth 2875.44, 2800.72
    #2649.47, 2623.07
    #2463.83, 2085.5
    
    
    
    
    
# ages, 102286, 95276
#       81816, 79905
#       63091, 38162

  


## depth_0_end    2696.3    index 350       ages  84962
## depth_1_start  2696.3          350       ages  84962
## depth_1_end    2231.03         779       ages  46717.



## depth_2_start  2231.03         779       ages  46717.
## depth_2_end    1381.3          1509      ages  9700


### ages
            

      

for i in range(len(Start)):
    fig,ax,twin2 = Make_fig()
    #fig,ax1,ax2,ax2v,ax3 = Make_fig2()
    start_year = Start[i]
    end_year = End[i]
    
    
    
    
    depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
    temp, temp_err = read_temp(data_path)
    d18O_full *= 1000
    acc = read_acc(data_path)
    depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full,ice_age_full,start_year, end_year)

    d18O_smooth = smooth_data(1 / 200., d18O_interval, ice_age_interval, ice_age_interval)[0]
    d15N_full = get_d15N_data_intervals(data_path)
    
    start_ind, end_ind = find_start_end_ind(ice_age_full, start_year, end_year)
    
    d15N_interval = d15N_full[start_ind: end_ind]


    temp_interval, temp_err_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year, end_year)
    acc_interval = get_interval_acc(acc, ice_age_full, start_year, end_year)
    
    

    # ages, 102286, 95276
    #       81816, 79905
    #       63091, 38162

      

                                                #age 122110    
    ## depth_0_end    2696.3    index 350       ages  84962
    ## depth_1_start  2696.3          350       ages  84962
    ## depth_1_end    2231.03         779       ages  46717.



    ## depth_2_start  2231.03         779       ages  46717.
    ## depth_2_end    1381.3          1509      ages  9700
    
    
    
    
    
    
   
    ind1 = 195
    ind2 = 227

    ind3 = 404
    ind4 = 411

    ind5 = 524
    ind6 = 911
    
    
    ind_0_end = 350
    ind_1_in = 351
    ind_1_end = 779
    
    ind_2_in = 780
    
    
    age_0_a = np.linspace(-122110,-102286,len(d15N[0:ind1]))
    age_0_b = np.linspace(-102286,-95276,len(d15N[ind1:ind2]))
    age_0_c = np.linspace(-95276,-84962,len(d15N[ind2:ind_0_end]))
    
    
    age_1_a = np.linspace(-84962,-81816,len(d15N[ind_1_in:ind3]))
    age_1_b = np.linspace(-81816,-79905,len(d15N[ind3:ind4]))
    age_1_c = np.linspace(-79905,-63091,len(d15N[ind4:ind5]))
    age_1_d = np.linspace(-63091,-46717,len(d15N[ind5:ind_1_end]))
    
    age_2_a = np.linspace(-46717,-38162,len(d15N[ind_2_in:ind6]))
    age_2_b = np.linspace(-38162,-9700,len(d15N[ind6:-1]))
    
    
    size = 2
    if i == 0:
        #ind1,ind2 = find_start_end_ind(ice_age_interval, -102524, -95765)
        #ice_age_in,ice_age_out = slice_time_series(ice_age, ind1, ind2)        
        #d15N_in,d15N_out = slice_time_series(d15N_interval, ind1, ind2)
        ax[1].plot(age_0_a,d15N[0:ind1],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_0_b,d15N[ind1:ind2],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_0_c,d15N[ind2:ind_0_end],'-',markersize=size,color='lightgrey')
        
        ax[1].plot(age_0_a,d15N[0:ind1],'o',markersize=size,color=color[0])
        ax[1].plot(age_0_b,d15N[ind1:ind2],'o',markersize=size,color=color[1])
        ax[1].plot(age_0_c,d15N[ind2:ind_0_end],'o',markersize=size,color=color[0])
        
        
    elif i == 1:
        #ind1,ind2 = find_start_end_ind(ice_age_interval, -82218, -80383)
        #ind3,ind4 = find_start_end_ind(ice_age_interval, -80162, -63986)
        ax[1].plot(age_1_a,d15N[ind_1_in:ind3],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_1_b,d15N[ind3:ind4],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_1_c,d15N[ind4:ind5],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_1_d,d15N[ind5:ind_1_end],'-',markersize=size,color='lightgrey')
        
        
        #ice_age_in,ice_age_out = slice_time_series(ice_age_interval, ind1, ind2)        
        #d15N_age_in,d15N_out = slice_time_series(d15N_interval_interval, ind1, ind2)
        ax[1].plot(age_1_a,d15N[ind_1_in:ind3],'o',markersize=size,color=color[0])
        ax[1].plot(age_1_b,d15N[ind3:ind4],'o',markersize=size,color=color[1])
        ax[1].plot(age_1_c,d15N[ind4:ind5],'o',markersize=size,color=color[0])
        ax[1].plot(age_1_d,d15N[ind5:ind_1_end],'o',markersize=size,color=color[2])
        
       
        

    elif i == 2:
        #ind1,ind2 = find_start_end_ind(ice_age_interval, -39746, -9961)
        ax[1].plot(age_2_a,d15N[ind_2_in:ind6],'-',markersize=size,color='lightgrey')
        ax[1].plot(age_2_b,d15N[ind6:-1],'-',markersize=size,color='lightgrey')
        
        ax[1].plot(age_2_a,d15N[ind_2_in:ind6],'o',markersize=size,color=color[2])
        ax[1].plot(age_2_b,d15N[ind6:-1],'o',markersize=size,color=color[3])
        
       
    fsize = 24
    #if i == 2:
    ax[2].set_xlabel('Age in years',fontsize=fsize)    
    
    ax[2].tick_params(axis='both', which='major', labelsize=20)
    ax[2].tick_params(axis='both', which='minor', labelsize=18)
    
    
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].tick_params(axis='both', which='minor', labelsize=18)
    
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[0].tick_params(axis='both', which='minor', labelsize=18)
    
    twin2.tick_params(axis='both', which='major', labelsize=20)
    twin2.tick_params(axis='both', which='minor', labelsize=18)
    
    
    
    #ax[0].set(xlabel=None)
    #ax[1].set(xlabel=None)
    #twin2.set(xlabel=None)
    ax[0].set_xlabel('')
    #ax[2].set_xlabel('')

    twin2.set_xlabel('')
    ax[1].set_xlabel('')
    
    #ax1.tick_params(axis='x', which='both', length=0)       
    
    ax[1].set_ylabel(u'$\delta^{15}$N  [‰]',fontsize=fsize)
    ax[2].set_ylabel(u'$\delta^{18}$O  [‰]',fontsize=fsize)
    ax[0].set_ylabel(u'Temp (\u00B0C)',fontsize=fsize)
    twin2.set_ylabel(r'Acc. [m/yr]',fontsize=fsize)
    
    
    
    
    ax[2].plot(ice_age_interval,d18O_interval,'ko',markersize=1)
    ax[2].plot(ice_age_interval,d18O_smooth,'r')
    ax[0].plot(ice_age_interval, temp_interval,'g')
    
    twin2.plot(ice_age_interval,acc_interval,'y')
        
    twin2.spines.right.set_bounds(acc_interval.min(), acc_interval.max())
    #twin1.spines.right.set_bounds(acc_interval.min(), acc_interval.max())

    ax[1].set_position((ax[2].get_position().x0, ax[2].get_position().y0 +0.2, 
                  ax[2].get_position().width, ax[2].get_position().height))
    ax[2].patch.set_alpha(0)
    ax[0].set_position((ax[1].get_position().x0, ax[1].get_position().y0 +0.21, 
                  ax[1].get_position().width, ax[1].get_position().height))
    
    plt.tight_layout()
    plt.savefig('NGRIP_data'+str(i)+'_1.png',dpi=600)
    
    #plt.clf()



















