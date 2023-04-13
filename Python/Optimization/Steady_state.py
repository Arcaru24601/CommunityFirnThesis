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
from Kindler_fit_ODR import input_file,expfunc,rho_0,rho_bco
import json
from pathlib import Path

M = 3

#Tem,Ac, Beta = input_file(num = 10)
np.random.seed(42)

#0.597, 0.463
#217.97, 229.96
Point_N = np.array([0.53,0.597,0.375,0.297])
Point_T = np.array([215.06,217.97,235,244.99])
Point_A = np.array([0.0284,0.0535,0.1607,0.2621])
s = np.random.normal(Point_N[M],0.02,size=50)
Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())]



def cost_func(d15N_ref,d15n_model):
    cost_fun = d15n_model - d15N_ref
    return cost_fun



model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5' ### Insert name
results_path = 'resultsFolder/Normal/minimizer.h5' #Add name


# =============================================================================
# Set parameters 
# =============================================================================

spin_year = 1000
model_year = 500
spin_year2 = spin_year + model_year/2
end_year = spin_year + model_year
stpsPerYear = 0.5
S_PER_YEAR = 60*60*24*365.25

#N_ref = Data_d15N.max() ##### Chosen from the 3sigma dist
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


'''
var_dict = {'count': np.zeros([N, 1], dtype=int),
            'd15N@CoD' : np.zeros([N, 1]),
            'temp' : np.zeros([N, 1]),
            'cost_func': np.zeros([N, 1])
            }
'''



def func(temp,N_ref,var_dict):
    
    count = int(np.max(var_dict['count']))
    print('Iteration',count)
    
    i_temp = np.full(len(Time),temp)
    input_temp = np.array([Time,i_temp])
    #i_acc = np.full(len(Time),expfunc(Beta, i_temp))
    #input_acc = np.array([Time, i_acc])
    
    os.chdir('../CFM/CFM_main')
    
    np.savetxt('CFMinput/OptiNoise/optimize_temp.csv', input_temp, delimiter=',')
    #np.savetxt('CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')
    subprocess.run('python main.py FirnAir_Noise.json -n')
    
    os.chdir('../../Optimization')
    #print(model_path)
    if os.path.exists(model_path):
        d15N_mode, temperature_model,d15N_diffu,CoD,Acc = get_model_data(model_path)
        d15N_model = d15N_mode
        print(d15N_mode,d15N_diffu)
        cost_fun = cost_func(N_ref, d15N_model)
        print('Temperature', temperature_model)
        print('d15N@CoD', d15N_model)
    
        var_dict['d15N@CoD'][count] = d15N_model
        var_dict['temp'][count] = temperature_model
        rho_co = rho_bco(temp)
        var_dict['rho_co'][count] = rho_co
        var_dict['CoD'][count] = CoD
        rho_s = rho_0(temperature_model, Acc)
        var_dict['rho_s'][count] = rho_s
    else:
        print('There is no output file -_- ')
        #os.chdir('../icecore_data/src/')
        cost_fun = 100.
        print('------------------------------------------------------------------------------------------')
        print('<<<<<<<< Close-off crashed - Setting cost function to 100! >>>>>>>>>>>>>>')
        print('------------------------------------------------------------------------------------------')
    
    var_dict['cost_func'][count] = cost_fun
    count += 1
    var_dict['count'][count] = count
    print('Cost func', cost_fun)
    #print('Acc',i_acc[0])
    
    return cost_fun
'''
print(N_ref)
res_c = brentq(func,a = 210,b = 260,args=(N_ref),full_output = True,xtol=2e-4,rtol=8.88e-8)
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
Dist_num =  ['Dist' + str(i) for i in np.arange(4)]### Regime number
Point_dist_num = np.arange(50)


def root_find(path_to_result,N_ref):
    count = 0
    
    var_dict = {'count': np.zeros([N, 1], dtype=int),
                'd15N@CoD' : np.zeros([N, 1]),
                'temp' : np.zeros([N, 1]),
                'cost_func': np.zeros([N, 1]),
                'CoD': np.zeros([N,1]),
                'rho_co': np.zeros([N,1]),
                'rho_s': np.zeros([N,1])
                }

    res_c = brentq(func,a = 213,b = 250,args=(N_ref,var_dict),full_output = True,xtol=2e-3,rtol=8.88e-6)
    entry_0 = np.where(var_dict['count'] == 0)[0]
    var_dict['count'] = np.delete(var_dict['count'], entry_0[1:])
    var_dict['count'] = var_dict['count'][:-1]
    max_int = np.shape(var_dict['count'])[0]
    print(res_c[1])

    with hf.File(path_to_result, 'w') as f:
        for key in var_dict:
            f[key] = var_dict[key][:max_int]
    f.close()
    var_dict.clear()
    return None    

for i in range(len(Point_N)):
    M = i
    s = np.random.normal(Point_N[M],0.02,size=50)
    Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())]
    bins2 = np.histogram_bin_edges(Data_d15N, bins='auto')
    n, bins, patches = plt.hist(x=Data_d15N, bins=bins2,alpha=1-0.1*i, rwidth=0.85)

import pandas as pd
#df = pd.read_csv('resultsFolder/out_model.csv',sep=',')
df = pd.read_csv('resultsFolder/out_model.csv',sep=',')

model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5' ### Insert name
#results_path = 'resultsFolder/minimizer.h5' #Add name
folder_path1 = 'resultsFolder/Normal/'
def Data_crunch(Model,Dist):
    for i in range(len(Model)):
        os.chdir('../CFM/CFM_main')
        file = open('FirnAir_Noise.json')
        data = json.load(file)
        data['physRho'] = str(Model[i])
            
        with open("FirnAir_Noise.json", 'w') as f:
            json.dump(data, f,indent = 2)
            
            # Closing file
        f.close()
        os.chdir('../../Optimization')
        for j in range(len(Dist)):
            
            s = np.random.normal(Point_N[j],0.02,size=1000)
            d15N_dist = s[(abs(s - s.mean())) < (3 * s.std())][:1]
            
            
            
            print(d15N_dist)
            for k in range(len(d15N_dist)):
                print(Model[i],Dist[j],k)
                d15N_ref = d15N_dist[k]
                folder_path = folder_path1 + str(Model[i]) + '/' + str(Dist[j]) 
                path = Path(folder_path)
                path.mkdir(parents=True, exist_ok=True)
                results_path = folder_path + '/' + 'Point'  + str(k) + '.h5'
                try:
                    root_find(results_path,d15N_ref)
                    
                except Exception as e: print(e)
                
    return results_path


Model_n = ['HLdynamic']
Dist_n = ['Dist_1']
Input_temp,Input_acc,Beta = input_file(num=25)
Indices_d15N = np.array([2,7,15])
#T = Data_crunch(Model_n, Dist_n)
Modela = ['HLD','HLS','BAR','GOU']
#Modela = #['Freitag', 'Schwander', 'Severinghaus', 'Witrant', 'Battle', 'Adolph']

def Data_crunch_Test(Model):
    for i in range(len(Model)):
        os.chdir('../')
        file = open('CFM/CFM_main/FirnAir_Noise.json')
        data = json.load(file)
        data['physRho'] = str(Model[i])
            
        with open("CFM/CFM_main/FirnAir_Noise.json", 'w') as f:
            json.dump(data, f,indent = 2)
            
            # Closing file
        f.close()
        
        #for j in range(len(Dist)):
            
            #s = np.random.normal(Point_N[j],0.02,size=1000)
            #d15N_dist = s[(abs(s - s.mean())) < (3 * s.std())][:1]
        
        
        for j in range(len(Indices_d15N)):
            mu_d15n = df[Modela[i]][Indices_d15N[j]]
            s = np.random.normal(np.asarray(mu_d15n),0.02,size=1000)
            d15N_dist = s[(abs(s - s.mean())) < (3 * s.std())][:50]
        
       
            
            print(d15N_dist)
            for k in range(len(d15N_dist)):
                print(Model[i],k)
                d15N_ref = mu_d15n#d15N_dist[k]
                print(k,Indices_d15N[j])
                print('Target d15N', d15N_ref)
                print('Target temp', Input_temp[Indices_d15N[j]])
            
                i_acc = np.full(len(Time),Input_acc[Indices_d15N[j]])
                input_acc = np.array([Time, i_acc])
            
            #D:\GitHub\CommunityFirnThesis\CommunityFirnThesis\Python\CFM\CFM_main\CFMinput\OptiNoise
                np.savetxt(r'D:/GitHub/CommunityFirnThesis/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')

                os.chdir(r'D:/GitHub/CommunityFirnThesis/CommunityFirnThesis/Python/Optimization')
            
            
                folder_path = folder_path1 + str(Model[i]) 
                path = Path(folder_path)
                path.mkdir(parents=True, exist_ok=True)
                results_path = folder_path + '/' + 'Point'  + str(k) + '.h5'
                try:
                    root_find(results_path,d15N_ref)
                    
                except Exception as e: print(e)
                
    return results_path,d15N_dist

#Diffus = ['Freitag', 'Schwander', 'Severinghaus', 'Witrant', 'Battle', 'Adolph']

T,d15N_dist = Data_crunch_Test(Model_name)









