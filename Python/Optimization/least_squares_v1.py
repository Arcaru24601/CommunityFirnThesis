# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:55:22 2023

@author: jespe
"""

from d18O_read import *
from read_d15n import *
from read_temp_acc import *
from calc_d15N import *
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import h5py as hf
import subprocess

def cost_func(ratio_model, ratio_data,ratio_err,length, mode='MSE'):
    len_opt = length
    cost_fun = 1 / (np.shape(ratio_model[-len_opt:])[0] -1) \
                * np.sum(((ratio_model[-len_opt:] - ratio_data[-len_opt:])
                          / ratio_err[-len_opt:]) ** 2)
                
    return cost_fun

data_path = 'data/NGRIP/Interpolated.xlsx'
data_path2 = 'data/NGRIP/supplement.xlsx'
model_path = '../CFM/CFM_main/CFMoutput/Optimization/CFMresults.hdf5.' ### Insert name
results_path = 'resultsFolder/minimizer.h5' #Add name

# =============================================================================
# Set parameters
# =============================================================================

start_year_ = -37000
end_year_ = -31000
stpsPerYear = 0.5
S_PER_YEAR = 60*60*24*365.25



cop_ = 1/200
time_grid_stp_ = 20
cod_mode = 'CoD'


optimizer = 'minimize' #least_squares
method = 'BFGS'
theta_ = [0.9,7]
theta_0 = [0.30, 7] # init guess
N = 1000 # Maximum amount of iterations

d15n_age = 'ice_age'
frac_minimizer_interval = 0.5



# =============================================================================
# Read data from NGRIP
# =============================================================================



depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full
                                                                , ice_age_full, start_year_, end_year_)


d18O_interval_perm = d18O_interval *1000
d18o_smooth = smooth_data(cop_, d18O_interval_perm, ice_age_interval, ice_age_interval)[0]

t1 = 1. / theta_[0] * d18o_smooth + theta_[1]
t = (d18o_smooth + 35.1)/(theta_0[0]) -31.55 + theta_0[1]




temp, temp_err = read_temp(data_path)
temp_interval,temp_err_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year_, end_year_)
### Add artificial for temp_err


#plt.plot(ice_age_interval,t)
#plt.plot(ice_age_interval,temp_interval)
#plt.plot(ice_age_interval,t_artificial)


acc = read_acc(data_path)
acc_interval = get_interval_acc(acc, ice_age_full, start_year_, end_year_)
input_acc = np.array([ice_age_interval,acc_interval])
np.savetxt('../CFM/CFM_main/CFMinput/Optimization/optimize_acc.csv', input_acc, delimiter=',')


years = (np.max(ice_age_interval) - np.min(ice_age_interval)) * 1.0
dt = S_PER_YEAR / stpsPerYear
stp = int(years * S_PER_YEAR / dt) #-1
modeltime = np.linspace(start_year_, end_year_, stp +1)[:-1]
#Test = int(np.shape(modeltime)[0] / 2)
minimizer_interval = int(np.shape(modeltime)[0] * frac_minimizer_interval)


var_dict = {'count': np.zeros([N, 1], dtype=int),
            'alpha': np.zeros([N, 1]),
            'beta': np.zeros([N, 1]),
            'gamma': np.zeros([N, 1]),
            'delta': np.zeros([N, 1]),
            'd15N@CoD' : np.zeros([N, minimizer_interval]),
            'ice_age' : np.zeros([N, minimizer_interval]),
            'gas_age' : np.zeros([N, minimizer_interval]),
            'cost_func': np.zeros([N, 1])
            }

def func(theta):
    count = int(np.max(var_dict['count']))
    print('Iteration',count)
    
    alpha = theta[0]
    beta = theta[1]

    temperature = 1./alpha * d18o_smooth + beta
    input_temp = np.array([ice_age_interval, temperature])
    
    
    np.savetxt('../CFM/CFM_main/CFMinput/Optimization/optimize_Temp.csv', input_temp, delimiter=',')
    os.chdir('../CFM/CFM_main')
    if count == 0:
        os.system('python main.py FirnAir_NGRIP.json -n')
    else:
        os.system('python main.py FirnAir_NGRIP.json')
    os.chdir('../../Optimization/')
    
    d15N_model, ice_age_model, gas_age_model, delta_age = get_d15N_model(model_path, 'CoD', FirnAir=True, cop=1/200.)
    ind = [2,3]
    if d15n_age == 'ice_age':
        d15N_data, d15N_data_err = [get_d15N_data_interval(data_path2, ice_age_model)[index] for index in ind]
        
    else:
        d15N_data, d15N_data_err = [get_d15N_data_interval(data_path2, gas_age_model)[index] for index in ind]
    cost_fun = cost_func(d15N_model, d15N_data, d15N_data_err,minimizer_interval)
    
    var_dict['alpha'][count] = alpha
    var_dict['beta'][count] = beta
    var_dict['d15N@CoD'][count, :] = d15N_model[-minimizer_interval:]
    var_dict['ice_age'][count, :] = ice_age_model[-minimizer_interval:]
    var_dict['gas_age'][count, :] = gas_age_model[-minimizer_interval:]
    var_dict['cost_func'][count] = cost_fun
    count += 1
    var_dict['count'][count] = count
    
    
    return cost_fun



# =============================================================================
# Minimize 
# =============================================================================


res_c = minimize(func, theta_, method=method)
entry_0 = np.where(var_dict['count'] == 0)[0]
var_dict['count'] = np.delete(var_dict['count'], entry_0[1:])
var_dict['count'] = var_dict['count'][:-1]
max_int = np.shape(var_dict['count'])[0]
with hf.File(results_path, 'w') as f:
    for key in var_dict:
        f[key] = var_dict[key][:max_int]
f.close()

theta_c_1 = res_c.x

print('----------------------------------------------')
print('|            INFO MINIMIZE                   |')
print('----------------------------------------------')
print(res_c.message)
print(res_c.success)
print('Theta1:', theta_c_1)
print('bla')


