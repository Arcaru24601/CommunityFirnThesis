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

data_path = 'data/NGRIP/interpolated.xlsx'
start_year = -120000
end_year = -10000
depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
temp, temp_err = read_temp(data_path)
acc = read_acc(data_path)
depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full,ice_age_full,start_year, end_year)
temp_interval, temp_err_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year, end_year)
acc_interval = get_interval_acc(acc, ice_age_full, start_year, end_year)


input_temp = np.array([ice_age_interval,temp_interval])
input_acc = np.array([ice_age_interval, acc_interval])
plt.close('all')




#plt.figure()
#plt.plot(temp_interval,acc_interval,'o')
import scipy.odr as OD

def expfunc(B,x):
   return np.exp(-B[0] + B[1]*x)
'''

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
ax.set_title(r'Using curve\_fit() to fit an exponential function')
ax.set_ylabel('y-Values')
ax.set_xlabel('x-Values')

ax.legend()
'''
def input_file(num,temp=temp_interval,acc=acc_interval):
    temp += 273.15
    x = temp
    y = acc

    func = OD.Model(expfunc)
    mydata = OD.Data(x, y, wd=1./np.power(3,2))
    myodr = OD.ODR(mydata, func, beta0=[21, 0.08])
    myoutput = myodr.run()
    myoutput.pprint()
    x2 = np.linspace(215,249,1000)
    y2 = expfunc(myoutput.beta,x2)
    
    output_temp = np.linspace(np.min(x2),np.max(x2),num)
    output_acc = np.linspace(np.min(y2),np.max(y2),num)
    return output_temp, output_acc, myoutput.beta





#Testing = input_file(10)
