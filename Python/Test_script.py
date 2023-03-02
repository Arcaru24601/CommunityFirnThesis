# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:46:42 2022

@author: jespe
"""

import json,subprocess



def Terminal_run(Folder,i,priority,):
    file = open('CFM/CFM_main/example_const.json')
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = 'CFMresults.hdf5'
    data['resultsFolder'] = 'CFMoutput/' + str(Folder[i])
    data['InputFileFolder'] = 'CFMinput/' + str(Folder[i])
    if priority == 'Temp':
        data['InputFileNameTemp'] = str(Folder[i])+'.csv'
        data['InputFileNamebdot'] = 'Acc_const.csv'
 
        
    elif priority == 'Acc':
        data['InputFileNamebdot'] = str(Folder[i])+'.csv'
        data['InputFileNameTemp'] = 'Temp_const.csv'  
        
        
    with open("CFM/CFM_main/example_const.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    
    
    subprocess.run('python main.py example_const.json -n', shell=True, cwd='CFM/CFM_main/')


def Config_Edit(Model,Folder,priority,effect,advec):
    
    if effect == 'grav':
        file = open('CFM/CFM_main/example_grav.json')    
    elif effect == 'full':
        file = open('CFM/CFM_main/example_Air.json')
    elif Model == 'HLdynamic':
        file = open('CFM/CFM_main/example_HLD.json')
    elif Model == 'Barnola1991':
        file = open('CFM/CFM_main/example_BAR.json')
    elif Model == 'Goujon2003':
        file = open('CFM/CFM_main/example_GOU.json')
    
    
    #elif Folder == 'Osc':
        #file = open('CFM/CFM_main/example_osc.json')
    
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = str(Folder) + str(Model) + str(advec) + str(effect) + '.hdf5'#'CFMresults.hdf5'
    data['resultsFolder'] = 'CFMoutput/DO_event/' + str(Folder) + '/' + str(Model) + '/' + str(advec) + '/' + str(effect)
    data['InputFileFolder'] = 'CFMinput/DO_event/' + str(Folder)
    data['physRho'] = str(Model)
    
    
    
    if priority == 'Temp':
        data['InputFileNameTemp'] = str(Folder)+'_Temp.csv'
        data['InputFileNamebdot'] = str(Folder)+'_acc.csv'
 
        
    elif priority == 'Acc':
        data['InputFileNamebdot'] = str(Folder)+'_acc.csv'
        data['InputFileNameTemp'] = str(Folder)+'_Temp.csv'  
        
    if effect == 'grav':  
        with open("CFM/CFM_main/example_grav.json", 'w') as f:
            json.dump(data, f,indent = 2)
    elif effect == 'full':
        with open("CFM/CFM_main/example_Air.json", 'w') as f:
            json.dump(data, f,indent = 2)
        # Closing file
    f.close()
    
def Air_Edit(Model,effect,advec):
    
    if Model == 'HLdynamic':
        file = open('CFM/CFM_main/Air_HLD.json')
    elif Model == 'Barnola1991':
        file = open('CFM/CFM_main/Air_BAR.json')
    elif Model == 'Goujon2003':
        file = open('CFM/CFM_main/Air_GOU.json')
        
    
    data = json.load(file)
    data['advection_type'] = str(advec)
    
    if effect =='full':
        data['gravity'] = 'on'
        data['thermal'] = 'on'
    elif effect == 'grav':
        data['thermal'] = 'off'
    
        

    with open("CFM/CFM_main/AirConfig.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()
def Terminal_run2(Model,Folder,Priority,effect,advec):
    Config_Edit(Model,Folder,Priority,effect,advec)
    Air_Edit(Model,effect,advec)
    if effect == 'grav':
        subprocess.run('python main.py example_grav.json -n', shell=True, cwd='CFM/CFM_main/')
    elif effect == 'full':
        subprocess.run('python main.py example_Air.json -n', shell=True, cwd='CFM/CFM_main/')
    
def Terminal_run_Models(Model,Folder,Priority,effect,advec):
    Config_Edit(Model,Folder,Priority,effect,advec)
    Air_Edit(Model,effect,advec)
    if Model == 'HLdynamic':
        subprocess.run('python main.py example_HLD.json -n', shell=True, cwd='CFM/CFM_main/')
    elif Model == 'Barnola1991':
        subprocess.run('python main.py example_BAR.json -n', shell=True, cwd='CFM/CFM_main/')
    elif Model == 'Goujon2003':
        subprocess.run('python main.py example_GOU.json -n', shell=True, cwd='CFM/CFM_main/')





def Terminal_run_Models(Model,Exp,Folder):
    file = open('CFM/CFM_main/example_equi.json')
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = str(Exp) + '_' + str(Model) + str(Folder) +  '.hdf5'
    data['resultsFolder'] = 'CFMoutput/Equi2/' + str(Exp) + '/' + str(Model) + '/' + str(Folder) 
    data['InputFileFolder'] = 'CFMinput/Equi2/' + str(Exp) + '/' + str(Folder)
    data['InputFileNameTemp'] = 'Temp.csv'
    data['InputFileNamebdot'] = 'Acc.csv'
    data['physRho'] = str(Model)
        
    with open("CFM/CFM_main/example_equi.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    
        
    subprocess.run('python main.py example_equi.json -n', shell=True, cwd='CFM/CFM_main/')
   
def Terminal_run_Amp(Model,Exp,Folder):
    file = open('CFM/CFM_main/example_equiAmp.json')
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = str(Exp) + '_' + str(Model) + str(Folder) +  '.hdf5'
    data['resultsFolder'] = 'CFMoutput/EquiAmp2/' + str(Exp) + '/' + str(Model) + '/' + str(Folder) 
    data['InputFileFolder'] = 'CFMinput/EquiAmp2/' + str(Exp) + '/' + str(Folder)
    data['InputFileNameTemp'] = 'Temp.csv'
    data['InputFileNamebdot'] = 'Acc.csv'
    data['physRho'] = str(Model)
        
    with open("CFM/CFM_main/example_equiAmp.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    
        
    subprocess.run('python main.py example_equiAmp.json -n', shell=True, cwd='CFM/CFM_main/')

def Terminal_run_Noise(Model,temp,acc):
    file = open('CFM/CFM_main/Firn_Noise.json')
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = str(Model) + str(temp)  + 'K.hdf5'
    data['resultsFolder'] = 'CFMoutput/Noise/Round5/' + str(Model) + '/' + str(temp) + 'K' 
    data['InputFileFolder'] = 'CFMinput/Noise/Round5/' + str(temp) + 'K'
    data['InputFileNameTemp'] = 'Temp.csv'
    data['InputFileNamebdot'] = 'Acc.csv'
    data['physRho'] = str(Model)
        
    with open("CFM/CFM_main/Firn_Noise.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    
    
    
    
    
    subprocess.run('python main.py Firn_Noise.json -n', shell=True, cwd='CFM/CFM_main/')








