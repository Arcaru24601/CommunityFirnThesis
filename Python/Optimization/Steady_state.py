# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:58:07 2023

@author: jespe
"""

import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
import os
from read_model import get_model_data
from scipy.optimize import brentq,least_squares
import h5py as hf
import subprocess
from Kindler_fit_ODR import input_file,expfunc

M = 0

Tem,Ac, Beta = input_file(num = 10)
np.random.seed(42)

#0.597, 0.463
#217.97, 229.96
Point_N = np.array([0.53,0.597,0.375,0.297])
Point_T = np.array([215.06,217.97,235,244.99])
#Point_A = np.array([0.0284,0.0535,0.1607,0.2621])
s = np.random.normal(Point_N[M],0.02,size=50)
Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())]



def cost_func(d15N_ref,d15n_model):
    cost_fun = d15n_model - d15N_ref
    return cost_fun



model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5.' ### Insert name
results_path = 'resultsFolder/minimizer.h5' #Add name


# =============================================================================
# Set parameters 
# =============================================================================

spin_year = 1000
model_year = 400
spin_year2 = spin_year + model_year/2
end_year = spin_year + model_year
stpsPerYear = 0.5
S_PER_YEAR = 60*60*24*365.25

N_ref = Data_d15N[0] ##### Chosen from the 3sigma dist
temp_0 = Point_T[M] ##### Good guess from the original temp data
#acc_0 = Point_A[M]

N = 100 #### Max iterations

# =============================================================================
# Create csv's
# =============================================================================




Time = np.array([spin_year,spin_year2,end_year])

Temp = np.full_like(Time,temp_0)
#Bdot = np.full(len(Time),acc_0)


Temp_csv = np.array([Time,Temp])
#Bdot_csv = np.array([Time,Bdot])
#os.chdir('../CFM/CFM_main/')
#np.savetxt('../CFM/CFM_main/CFMinput/OptiNoise/optimize_acc.csv',Bdot_csv,delimiter=',')
np.savetxt('../CFM/CFM_main/CFMinput/OptiNoise/optimize_temp.csv',Temp_csv,delimiter=',')



var_dict = {'count': np.zeros([N, 1], dtype=int),
            'd15N@CoD' : np.zeros([N, 1]),
            'temp' : np.zeros([N, 1]),
            'cost_func': np.zeros([N, 1])
            }




def func(temp):
    count = int(np.max(var_dict['count']))
    print('Iteration',count)
    
    i_temp = np.full_like(Time,temp)
    input_temp = np.array([Time,i_temp])
    i_acc = np.full(len(Time),expfunc(Beta, i_temp))
    input_acc = np.array([Time, i_acc])
    
    os.chdir('../CFM/CFM_main')

    np.savetxt('CFMinput/OptiNoise/optimize_temp.csv', input_temp, delimiter=',')
    np.savetxt('CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')
    subprocess.run('python main.py FirnAir_Noise.json -n')
    
    os.chdir('../../Optimization')
    d15N_model, temperature_model,d15N_diffu = get_model_data(model_path)
    cost_fun = cost_func(N_ref, d15N_model)
    
    var_dict['d15N@CoD'][count] = d15N_model
    var_dict['temp'][count] = temperature_model
    var_dict['cost_func'][count] = cost_fun
    count += 1
    var_dict['count'][count] = count
    
    print('Cost func', cost_fun)
    print('Temperature', temperature_model)
    print('Acc',i_acc[0])
    print('d15N@CoD', d15N_model)
    return cost_fun


res_c = brentq(func,a = 213,b = 250,full_output = True,xtol=2e-8,rtol=8.88e-12)
entry_0 = np.where(var_dict['count'] == 0)[0]
var_dict['count'] = np.delete(var_dict['count'], entry_0[1:])
var_dict['count'] = var_dict['count'][:-1]
max_int = np.shape(var_dict['count'])[0]

with hf.File(results_path, 'w') as f:
    for key in var_dict:
        f[key] = var_dict[key][:max_int]
f.close()

temp_1 = res_c[0]


print('----------------------------------------------')
print('|            INFO MINIMIZE                   |')
print('----------------------------------------------')
print(res_c[1])
print('Temp1:', temp_1)
print('bla')



'''
res_c = least_squares(func,x0=215,bounds=(209,265))
entry_0 = np.where(var_dict['count'] == 0)[0]
var_dict['count'] = np.delete(var_dict['count'], entry_0[1:])
var_dict['count'] = var_dict['count'][:-1]
max_int = np.shape(var_dict['count'])[0]

with hf.File(results_path, 'w') as f:
    for key in var_dict:
        f[key] = var_dict[key][:max_int]
f.close()

temp_1 = res_c.x


print('----------------------------------------------')
print('|            INFO MINIMIZE                   |')
print('----------------------------------------------')
print(res_c.message)
print(res_c.success)
print('Temp1:', temp_1)
print('bla')
'''


Model_name = ['HLdynamic','HLSigfus','Barnola1991','Goujon2003']
Dist_num =  ['Dist' + str(i) for i in np.arange(1,5)]### Regime number
Point_dist_num = np.arange(50)


def root_find(path_to_result,N_ref):
    res_c = brentq(func,a = 213,b = 250,args=(N_ref),full_output = True,xtol=2e-8,rtol=8.88e-12)
    entry_0 = np.where(var_dict['count'] == 0)[0]
    var_dict['count'] = np.delete(var_dict['count'], entry_0[1:])
    var_dict['count'] = var_dict['count'][:-1]
    max_int = np.shape(var_dict['count'])[0]

    with hf.File(path_to_result, 'w') as f:
        for key in var_dict:
            f[key] = var_dict[key][:max_int]
    f.close()
        

for i in range(len(Point_N)):
    M = i
    s = np.random.normal(Point_N[M],0.02,size=50)
    Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())]
    bins2 = np.histogram_bin_edges(Data_d15N, bins='auto')
    n, bins, patches = plt.hist(x=Data_d15N, bins=bins2,alpha=1-0.1*i, rwidth=0.85)



model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5.' ### Insert name
results_path = 'resultsFolder/minimizer.h5' #Add name

def Data_crunch(Model,Dist):
    for i in range(len(Model)):
        for j in range(len(Dist)):
            
            s = np.random.normal(Point_N[M],0.02,size=1000)
            d15N_dist = s[(abs(s - s.mean())) < (3 * s.std())][:50]
            
            for k in range(len(d15N_dist)):
                print(Model[i],Dist[j],k)
                try:
                    d15N_ref = d15N_dist[k]
                    results_path = str(Model[i]) + '_'  + str(Dist[j]) + '_Point'  + str(k) + '.h5'
                    root_find(results_path,d15N_ref)
                    
                except Exception as e: print(e)
    return None



















