# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:37:11 2023

@author: Jesper Holm
"""

import numpy as np
from read_temp_acc import *
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

data_path = 'data/NGRIP/interpolated.xlsx'
start_year = -120000
end_year = -10000
depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
temp, temp_err = read_temp(data_path)
acc = read_acc(data_path)
depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full,ice_age_full,start_year, end_year)
temp_interval, temp_err_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year, end_year)
acc_interval = get_interval_acc(acc, ice_age_full, start_year, end_year)

temp_interval += 273.15
input_temp = np.array([ice_age_interval,temp_interval])
input_acc = np.array([ice_age_interval, acc_interval])
plt.close('all')
import scipy.odr as OD

def expfunc(B,x):
   return np.exp(-B[0] + B[1]*x)
def fun(T):
    y = np.exp(-21.492 + 0.0811 * T)
    return y

func = OD.Model(expfunc)
mydata = OD.Data(temp_interval, acc_interval, wd=1./np.power(3,2))
myodr = OD.ODR(mydata, func, beta0=[21, 0.08])
myoutput = myodr.run()
myoutput.pprint()
Test = np.linspace(np.min(temp_interval),np.max(temp_interval),1000)
y = expfunc(myoutput.beta,Test)

fig, ax = plt.subplots()
ax.scatter(temp_interval, acc_interval, label='Raw data')

ax.plot(Test,y,'k',label = 'Func')
ax.plot(Test,fun(Test),'r-',label='gkinis')
ax.set_title(r'Using curve\_fit() to fit an exponential function')
ax.set_ylabel('y-Values')
ax.set_xlabel('x-Values')

ax.legend()

