# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:37:12 2023

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


def input_file(num,temp=temp_interval,acc=acc_interval):

    x = temp
    y = acc

    popt, pcov = curve_fit(lambda t, a, b, c, d: a * np.exp(b * t + c) + d, x,y)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]

    x2 = np.linspace(213-273.15,250-273.15,1000)
    y2 = a * np.exp(b * x2 + c) + d
    #X2 = x2 + 273.15
    
    output_temp = np.linspace(np.min(x2),np.max(x2),num)
    output_acc = np.linspace(np.min(y2),np.max(y2),num)
    return output_temp+273.15, output_acc,a,b,c,d

def func(T):
    y = np.exp(-21.492 + 0.0811 * T)
    return y

    
#Y = func(X_fitted_Kelvin)
#Y2 = func(X2)
x = temp_interval+273.15
y = acc_interval

popt, pcov = curve_fit(lambda t, a, b: np.exp(-a * t + b), x,y)
a = popt[0]
b = popt[1]
#c = popt[2]
#d = popt[3]

X2 = np.linspace(215-273.15,250-273.15,1000)
y2 = np.exp(-a * X2 + b)
#X1 = X2 + 273.15
#Y2 = func(X1)

fig, ax = plt.subplots()
ax.scatter(x, y, label='Raw data')
#ax.plot(X_fitted_Kelvin, y_fitted, 'k', label='Fitted curve')
#ax.plot(x_fitted,p(x_fitted),'r',label='Poly')
ax.plot(X2+273.15,y2,'k',label = 'Func')
#ax.plot(X2,Y2,'r',label = 'Art')
ax.set_title(r'Using curve\_fit() to fit an exponential function')
ax.set_ylabel('y-Values')
#ax.set_ylim(0, 500)
ax.set_xlabel('x-Values')
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.legend()











