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
from Kindler_fit_Clear import input_file,expfunc,rho_0,rho_bco,get_min_max_temp
import json
from pathlib import Path
import pandas as pd


np.random.seed(42)


plt.close('all')


def cost_func(d15N_ref,d15n_model):
    cost_fun = d15n_model - d15N_ref
    return cost_fun



 ### Insert name


# =============================================================================
# Set parameters 
# =============================================================================

spin_year = 1000
model_year = 650
spin_year2 = spin_year + model_year/2
end_year = spin_year + model_year



N = 100 #### Max iterations

# =============================================================================
# Create csv's
# =============================================================================


Time = np.array([spin_year,spin_year2,end_year])

df = pd.read_csv('resultsFolder/Integer_diffu.csv',sep=',')
Input_temp,Input_acc,Beta = input_file()

def reset_to_factory(file_id):
    
    file = open(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/Air_OptiNoise_Ulti_'+str(file_id)+'.json')
    data = json.load(file)
    data['Diffu_param'] = 'Schwander'
    data['noisy_bco'] = False
    with open(r"/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/Air_OptiNoise_Ulti_"+str(file_id)+".json", 'w') as f:
        json.dump(data, f,indent = 2)
    
    # Closing file
    f.close()

    file = open(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/FirnAir_Noise_Ulti_'+str(file_id)+'.json')
    data = json.load(file)
    data['rhos0'] = 350.0
    data['resultsFolder'] = "CFMoutput/OptiNoise/" + str(file_id)
    data['InputFileFolder'] = "CFMinput/OptiNoise/" + str(file_id)
    data['AirConfigName'] = "Air_OptiNoise_Ulti_"+str(file_id)+".json"

    with open(r"/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/FirnAir_Noise_Ulti_"+str(file_id)+".json", 'w') as f:
        json.dump(data, f,indent = 2)

    # Closing file
    f.close()
    return None

def func(temp,N_ref,var_dict,bco_param_flag,file_id):
    
    count = int(np.max(var_dict['count']))
    print('Iteration',count)
    file = open(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/FirnAir_Noise_Ulti_'+str(file_id)+'.json')
    data = json.load(file)
    rhos = data['rhos0']

    file.close() 
    #print('Using',rhos,'surface density')
  
  
    
    i_temp = np.full(len(Time),temp)
    input_temp = np.array([Time,i_temp])
    np.savetxt(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/'+str(file_id)+'/optimize_temp.csv', input_temp, delimiter=',')
    
    
    os.chdir('../CFM/CFM_main')
    
    #np.savetxt('CFMinput/OptiNoise/optimize_acc.csv', input_acc, delimiter=',')
    
    call_args = ['python', 'main.py', 'FirnAir_Noise_Ulti_'+str(file_id)+'.json', '-n']
    subprocess.run(call_args)
    
    os.chdir('../../Optimization')
    #print(model_path)
    model_path = '../CFM/CFM_main/CFMoutput/OptiNoise/'+str(file_id)+'/CFMresults.hdf5'
    if os.path.exists(model_path):
        d15N_mode, temperature_model,d15N_diffu, CoD,Acc,diffu = get_model_data(model_path)
        d15N_model = d15N_diffu
        print(d15N_mode,d15N_diffu)
        cost_fun = cost_func(N_ref, d15N_model)
        print('Temperature', temp)
        print('Model rho',rhos)
        print('d15N@CoD', d15N_model)
    
        var_dict['d15N@CoD'][count] = d15N_model
        var_dict['temp'][count] = temperature_model
        var_dict['diffusivity'][count] = diffu
        
        var_dict['d15N@CoD'][count] = d15N_model
        var_dict['temp'][count] = temperature_model
        if bco_param_flag == True:
            file = open(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/Air_OptiNoise_Ulti_'+str(file_id)+'.json')
            data = json.load(file)
            rho_co = data['bco_dist'] 
            file.close() 
            #print('Using',rhos,'surface density')
        else:
            rho_co = rho_bco(temperature_model)
        var_dict['rho_co'][count] = rho_co
        var_dict['CoD'][count] = CoD
        #rho_s = rho_0(temperature_model, Acc)
        var_dict['rho_s'][count] = rhos
        
        
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




def root_find(path_to_result,N_ref,bco_param_flag,file_id):
    count = 0
    
    var_dict = {'count': np.zeros([N, 1], dtype=int),
                'd15N@CoD' : np.zeros([N, 1]),
                'temp' : np.zeros([N, 1]),
                'cost_func': np.zeros([N, 1]),
                'CoD': np.zeros([N,1]),
                'rho_co': np.zeros([N,1]),
                'rho_s': np.zeros([N,1]),
                'diffusivity': np.zeros([N,1])
                }

    res_c = brentq(func,a = 215,b = 255,args=(N_ref,var_dict,bco_param_flag,file_id),full_output = True,xtol=2e-3,rtol=8.88e-6)
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

temp_NGRIP_min, temp_NGRIP_max = get_min_max_temp()
temp_test = np.linspace(temp_NGRIP_min,temp_NGRIP_max,1000)+273.15
bco = rho_bco(temp_test)

Temp_input = np.array([1,2,3,4,5,6])

def Data_crunch_Ulti(sath,N,rho_surface_uncertainty_Flag,diff_param_Flag,bco_param_flag,file_id,Model='HLD',Indices=Temp_input):
    print(os.getcwd())

    os.chdir('../')
  
        
    # =============================================================================
    #    Generate d15N distribution 
    # =============================================================================
        
    for j,val in enumerate(Indices):
        mu_d15n = df[str(Model)][Indices[j]]
        d = np.random.normal(np.asarray(mu_d15n),0.02,size=2000)
        d15N_dist = d[(abs(d - d.mean())) < (3 * d.std())][:N]
 
        print(mu_d15n,Input_temp[Indices[j]],Input_acc[Indices[j]],expfunc(Beta,Input_temp[Indices[j]]))    
        rho_s_distribution = np.random.normal(330,20,size=N)
        bco_distribution = np.random.normal(np.mean(bco),15,size=N)

            
            
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
                
                file = open('CFM/CFM_main/FirnAir_Noise_Ulti_'+str(file_id)+'.json')
                data = json.load(file)
                data['rhos0'] = rho_s_distribution[k]
            
                with open("CFM/CFM_main/FirnAir_Noise_Ulti_"+str(file_id)+".json", 'w') as f:
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
                
                file = open('CFM/CFM_main/Air_OptiNoise_Ulti_'+str(file_id)+'.json')
                data = json.load(file)
                data['Diffu_param'] = str(diff_param)
                    
                with open("CFM/CFM_main/Air_OptiNoise_Ulti_"+str(file_id)+".json", 'w') as f:
                    json.dump(data, f,indent = 2)
                    
                    # Closing file
                f.close()
                
            
                print('Using',diff_param,'diffusion parameterization')
            
            # =============================================================================
            #   Close off density          
            # =============================================================================
            if bco_param_flag == True:
                bco_dist = np.random.choice(bco_distribution)
                #print(bco_d)
                file = open('CFM/CFM_main/Air_OptiNoise_Ulti_'+str(file_id)+'.json')
                data = json.load(file)
                data['bco_dist'] = bco_dist
                data['noisy_bco'] = True
                #print(bco_dist)
                with open("CFM/CFM_main/Air_OptiNoise_Ulti_"+str(file_id)+".json", 'w') as f:
                    json.dump(data, f,indent = 2)
                    
                    # Closing file
                f.close()
                
            
                print('Using',str(bco_dist),'close off density')
            
            

            
            
            # =============================================================================
            #    Generate accumulation file              
            # =============================================================================
            
            
            i_acc = np.full(len(Time),Input_acc[Indices[j]])
            input_acc = np.array([Time, i_acc])
            
            np.savetxt(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/' + str(file_id)+'/optimize_acc.csv', input_acc, delimiter=',')
                
            os.chdir(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/Optimization')
            
            folder_path = str(sath) + str(Model) + '/' + str(Indices[j]) 
            
            path = Path(folder_path)
            path.mkdir(parents=True, exist_ok=True)
            results_path = folder_path + '/' + 'Point'  + str(k) + '.h5'
            try:
                root_find(results_path,d15N_ref,bco_param_flag,file_id)
                    
            except Exception as e: print(e)
                
    return None
path_Temp = 'resultsFolder/Ulti_Temp/'
path_rho = 'resultsFolder/Ulti_rho/'
path_Deff = 'resultsFolder/Ulti_Deff/'
path_bco = 'resultsFolder/Ulti_Bco/'

path_bco_rho = 'resultsFolder/Ulti_bco_rho/'
path_Deff_only = 'resultsFolder/Ulti_Deff_only/'
path_bco_only = 'resultsFolder/Ulti_bco_only/'

#for j,val in enumerate(Input_temp):

print(Input_temp)
m = 800# Num repetitions

import time
#t1 = time.time()
#Data_crunch_Ulti(path_Temp,m,False,False,False,file_id=0)
#Data_crunch_Ulti(path_rho,m,True,False,False,file_id=1)
#Data_crunch_Ulti(path_bco_only,m,False,False,True,file_id=2)
#t2 = time.time()
#print(t2-t1)


Input_2 = np.array([1])    
Input_3 = np.array([2])
Input_4 = np.array([3])
Input_5 = np.array([4])
Input_6 = np.array([5])





from multiprocessing import Process

for g in range(3):
	reset_to_factory(g)
	paths = Path(r'/home/jesperholm/Documents/GitHub/CommunityFirnThesis/Python/CFM/CFM_main/CFMinput/OptiNoise/' + str(g))
	paths.mkdir(parents=True, exist_ok=True)

'''
if __name__ == "__main__":
    # construct a different process for each function
    ti = time.time()
    processes = [Process(target=Data_crunch_Ulti, args=(path_Temp,m,False,False,False,0)),
                 Process(target=Data_crunch_Ulti, args=(path_rho,m,True,False,False,1)),
                 Process(target=Data_crunch_Ulti, args=(path_bco_only,m,False,False,True,2))]

    # kick them off 
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()
    tf = time.time()
#print(t2-t1,'Serial')
    print(tf-ti,'Parallel')
'''



if __name__ == "__main__":
    # construct a different process for each function
    ti = time.time()
    processes = [Process(target=Data_crunch_Ulti, args=(path_Deff_only,m,False,True,False,0,2e-3,'HLD',Input_2)),
                 #Process(target=Data_crunch_Ulti, args=(path_Deff_only,m,False,True,False,1,2e-3,'HLD',Input_3)),
                 #Process(target=Data_crunch_Ulti, args=(path_Deff_only,m,False,True,False,2,1e-6,'HLD',Input_4)),
                 Process(target=Data_crunch_Ulti, args=(path_Deff_only,m,False,True,False,3,2e-3,'HLD',Input_5)),
                 Process(target=Data_crunch_Ulti, args=(path_Deff_only,m,False,True,False,4,2e-3,'HLD',Input_6))]
    # kick them off 
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()
    tf = time.time()
#print(t2-t1,'Serial')
    print(tf-ti,'Parallel')




