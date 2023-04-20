# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:29:01 2023

@author: Jesper Holm
"""


import numpy as np
from matplotlib import pyplot as plt
import random
import os
from read_model import get_model_data
from scipy.optimize import brentq
import h5py as hf
import subprocess
from Kindler_fit_Clear import input_file,expfunc,rho_0,rho_bco
import json
from pathlib import Path
import pandas as pd


np.random.seed(42)


plt.close('all')


def cost_func(d15N_ref,d15n_model):
    cost_fun = d15n_model - d15N_ref
    return cost_fun



model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/CFMresults.hdf5' ### Insert name


# =============================================================================
# Set parameters 
# =============================================================================

spin_year = 1000
model_year = 500
spin_year2 = spin_year + model_year/2
end_year = spin_year + model_year



N = 100 #### Max iterations

# =============================================================================
# Create csv's
# =============================================================================


Time = np.array([spin_year,spin_year2,end_year])

df = pd.read_csv('resultsFolder/Integer_diffu.csv',sep=',')
Input_temp,Input_acc,Beta = input_file()



def func(temp,N_ref,var_dict,rhos):
    
    count = int(np.max(var_dict['count']))
    print('Iteration',count)
    
  
    
    i_temp = np.full(len(Time),temp)
    input_temp = np.array([Time,i_temp])
    np.savetxt(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/optimize_temp.csv', input_temp, delimiter=',')
    
    
    os.chdir('../CFM/CFM_main')
    
    #np.savetxt('CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')
    
    call_args = ['python', 'main.py', 'FirnAir_Noise_Ulti.json', '-n']
    subprocess.run(call_args)
    
    os.chdir('../../Optimization')
    #print(model_path)
    if os.path.exists(model_path):
        d15N_mode, temperature_model,d15N_diffu, CoD,Acc = get_model_data(model_path)
        d15N_model = d15N_diffu
        print(d15N_mode,d15N_diffu)
        cost_fun = cost_func(N_ref, d15N_model)
        print('Temperature', temp)
        print('Model rho',rhos)
        print('d15N@CoD', d15N_model)
    
        var_dict['d15N@CoD'][count] = d15N_model
        var_dict['temp'][count] = temperature_model
        
        
        var_dict['d15N@CoD'][count] = d15N_model
        var_dict['temp'][count] = temperature_model
        rho_co = rho_bco(temperature_model)
        var_dict['rho_co'][count] = rho_co
        var_dict['CoD'][count] = CoD
        #rho_s = rho_0(temperature_model, Acc)
        var_dict['rho_s'][count] = rho_0(temp,Acc)
        
        
    else:
        print('There is no output file -_- ')
        cost_fun = 100.
        print('------------------------------------------------------------------------------------------')
        print('<<<<<<<< Close-off crashed - Setting cost function to 100! >>>>>>>>>>>>>>')
        print('------------------------------------------------------------------------------------------')
    
    var_dict['cost_func'][count] = cost_fun
    count += 1
    var_dict['count'][count] = count
    print('Cost func', cost_fun)

    return cost_fun




def root_find(path_to_result,N_ref,rhos):
    count = 0
    
    var_dict = {'count': np.zeros([N, 1], dtype=int),
                'd15N@CoD' : np.zeros([N, 1]),
                'temp' : np.zeros([N, 1]),
                'cost_func': np.zeros([N, 1]),
                'CoD': np.zeros([N,1]),
                'rho_co': np.zeros([N,1]),
                'rho_s': np.zeros([N,1])
                }

    res_c = brentq(func,a = 215,b = 250,args=(N_ref,var_dict,rhos),full_output = True,xtol=2e-3,rtol=8.88e-6)
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



def Data_crunch_Ulti(Model,Indices,sath,N,rho_surface_uncertainty_Flag,diff_param_Flag):
    print(os.getcwd())

    os.chdir('../')
    #file = open('CFM/CFM_main/FirnAir_Noise_Ulti.json')
    #data = json.load(file)
    #data['physRho'] = str(Model)
            
    #with open("CFM/CFM_main/FirnAir_Noise_Ulti.json", 'w') as f:
    #    json.dump(data, f,indent = 2)
            
            # Closing file
    #f.close()
        
    # =============================================================================
    #    Generate d15N distribution 
    # =============================================================================
        
    for j,val in enumerate(Indices):
        mu_d15n = df[str(Model)][Indices[j]]
        d15N_dist = np.random.normal(np.asarray(mu_d15n),0.02,size=N)
            
        
           
            
            
        print(d15N_dist)
        for k in range(len(d15N_dist)):
            d15N_ref = d15N_dist[k]
            
            print(Model,k,Indices[j])
            print('Target d15N', d15N_ref)
            print('Target temp', Input_temp[Indices[j]])

            os.chdir(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/')
            
            # =============================================================================
            #     Add uncertainty in surface density
            # =============================================================================
            
            if rho_surface_uncertainty_Flag == True:
                rho_s_distribution = np.random.normal(330,20,size=N)
                file = open('CFM/CFM_main/FirnAir_Noise_Ulti.json')
                data = json.load(file)
                data['rhos0'] = rho_s_distribution[k]
            
                with open("CFM/CFM_main/FirnAir_Noise_Ulti.json", 'w') as f:
                    json.dump(data, f,indent = 2)
                
                    # Closing file
                f.close()
                print('Using',rho_s_distribution[k],'surface density')

            
            # =============================================================================
            #     Add uncertainty in diffusivity parameterization
            # =============================================================================
            
            if diff_param_Flag == True:
                diff = ['Freitag', 'Schwander', 'Severinghaus', 'Witrant', 'Battle', 'Adolph']
                diff_param = random.choice(diff)
                
                file = open('CFM/CFM_main/Air_OptiNoise_Ulti.json')
                data = json.load(file)
                data['Diffu_param'] = str(diff_param)
                    
                with open("CFM/CFM_main/Air_OptiNoise_Ulti.json", 'w') as f:
                    json.dump(data, f,indent = 2)
                    
                    # Closing file
                f.close()
                
            
                print('Using',diff_param,'diffusion parameterization')
            
                
            
            # =============================================================================
            #    Generate accumulation file              
            # =============================================================================
            i_acc = np.full(len(Time),Input_acc[Indices[j]])
            input_acc = np.array([Time, i_acc])
            
            np.savetxt(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')
                
            os.chdir(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/Optimization')
            
            folder_path = str(sath) + str(Model) + '/' + str(Indices[j]) 
            
            path = Path(folder_path)
            path.mkdir(parents=True, exist_ok=True)
            results_path = folder_path + '/' + 'Point'  + str(k) + '.h5'
            try:
                root_find(results_path,d15N_ref,rho_s_distribution[k])
                    
            except Exception as e: print(e)
                
    return None
path1 = 'resultsFolder/Ulti1/'
path2 = 'resultsFolder/Ulti2/'
#for j,val in enumerate(Input_temp):
Temp_input = np.array([1,3,5])
Data_crunch_Ulti('HLD',Temp_input,path1,750,False,False)
Data_crunch_Ulti('HLD',Temp_input,path1,750,True,False)
Data_crunch_Ulti('HLD',Temp_input,path2,750,True,True)
#rint(Time)


