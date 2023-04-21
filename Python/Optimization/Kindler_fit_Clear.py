# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:42:07 2023

@author: Jesper Holm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:58:25 2023

@author: jespe
"""

import numpy as np
from read_temp_acc import *
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
sns.set_theme()
data_path = 'data/NGRIP/interpolated.xlsx'
start_year = -120000
end_year = -10000
depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
temp, temp_err = read_temp(data_path)
acc = read_acc(data_path)
depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full,ice_age_full,start_year, end_year)
temp_interval, temp_err_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year, end_year)
acc_interval = get_interval_acc(acc, ice_age_full, start_year, end_year)

'''
plt.close('all')

input_temp = np.array([ice_age_interval,temp_interval])
input_acc = np.array([ice_age_interval, acc_interval])
fig,ax = plt.subplots(2,sharex=True)
ax[0].plot(ice_age_interval,temp_interval,label='Temperature')
ax[1].plot(ice_age_interval,acc_interval,'g',label='Accumulation')
handles, labels = [(a + b) for a, b in zip(ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels())]
#plt.gca().get_legend_handles_labels()
ax[0].set_ylabel('Temperature [Celsius]')
ax[1].set_ylabel('Accumulation rate [Unit]')
ax[1].set_xlabel('Ice Age/-x years ago [year]')
fig.legend(handles, labels, loc='upper center',ncol=2)
plt.savefig('Data.png',dpi=300)

'''

#plt.figure()
#plt.plot(temp_interval,acc_interval,'o')
import scipy.odr as OD

def expfunc(B,x):
   return np.exp(-B[0] + B[1]*x)
'''
temp_interval += 273.15
func = OD.Model(expfunc)
mydata = OD.Data(temp_interval, acc_interval, wd=1./np.power(3,2))
myodr = OD.ODR(mydata, func, beta0=[21, 0.08])
myoutput = myodr.run()
myoutput.pprint()
Test = np.linspace(np.min(temp_interval),np.max(temp_interval),1000)
y = expfunc(myoutput.beta,Test)


output_temp = np.linspace(np.min(Test),np.max(Test),10)
output_acc = expfunc(myoutput.beta,output_temp)

alpha,beta = myoutput.beta

fig, ax = plt.subplots()
ax.scatter(temp_interval, acc_interval, label='Raw data')
ax.plot(output_temp,output_acc,'ro',label='Forward points')
ax.plot(Test,y,'k',label = r'Fitted function' + r' $\exp( -{0:.2f} + {1:.2f}\cdot T)$'.format(alpha,beta))
#ax.set_title(r'Using curve\_fit() to fit an exponential function')
ax.set_ylabel(r'Accumulation rate [m yr$^{-1}$]',fontsize=14)
ax.set_xlabel('Temperature [K]',fontsize=14)
ax.legend(fontsize=14)
plt.savefig('Noise/TEst2.png',dpi=300)

'''
def input_file(temp=temp_interval,acc=acc_interval):
    temp += 273.15
    x = temp
    y = acc

    func = OD.Model(expfunc)
    mydata = OD.Data(x, y, wd=1./np.power(3,2))
    myodr = OD.ODR(mydata, func, beta0=[21, 0.08])
    myoutput = myodr.run()
    myoutput.pprint()
    x2 = np.linspace(np.min(x),np.max(x),1000)
    y2 = expfunc(myoutput.beta,x2)
    
    output_temp = np.array([215,220,225,230,235,240,245])#np.linspace(np.min(x2),np.max(x2),num)
    output_acc = expfunc(myoutput.beta,output_temp)
    temp -= 273.15
    return output_temp, output_acc, myoutput.beta

def rho_0(T,A,W=2.31):
    rho0 = 7.36e-2 + 1.06e-3 * T + 6.69e-2 * A + 4.77e-3*W
    return rho0*1000

def rho_bco(T):
    rho_co = 1 / (1 / (917.0) + T * 6.95E-7 - 4.3e-5)
    return rho_co

def get_min_max_temp(x=temp_interval):
    return np.min(x),np.max(x)



#Input_temp,Input_acc,Beta = input_file(25)
#rho_s = rho_0(Input_temp,Input_acc)
#rho_co = rho_bco(Input_temp)

'''
plt.close('all')
output_temp,output_acc,beta = input_file(num=25)

fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(temp_interval+273.15, acc_interval, label='Raw data')
ax.plot(output_temp,output_acc,'ro',label='Forward points')
ax.plot(output_temp,output_acc,'k',label = r' $\exp( -{0:.2f} + {1:.2f}\cdot T)$'.format(beta[0],beta[1]))
#ax.set_title(r'Using curve\_fit() to fit an exponential function')
ax.set_ylabel(r'Accumulation rate [m yr$^{-1}$]',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
#ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xlabel('Temperature [K]',fontsize=18)
ax.legend(fontsize=12)
plt.savefig('Noise/TEst2.png',dpi=300)
'''



#Testing = input_file(10)
