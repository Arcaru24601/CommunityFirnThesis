# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:46:42 2022

@author: jespe
"""
path = 'CFM/CFM_main/'

import os,subprocess,sys,json


def Terminal_run(Experiments_T,Experiments_A,i,priority):
    file = open(path+'example.json')
        
    data = json.load(file)
    if priority == 'Temp':
        data['InputFileFolder'] = 'CFMinput/' + str(Experiments_T[i])
        data['resultsFolder'] = 'CFMoutput/' + str(Experiments_T[i])
        data['InputFileNameTemp'] = 'Temp_' + str(Experiments_T[i]) + '.csv'
        data['InputFileNamebdot'] = 'Acc_const.csv'
        
    elif priority == 'Acc':
        data['InputFileFolder'] = 'CFMinput_' + str(Experiments_A[i])
        data['resultsFolder'] = 'CFMoutput_' + str(Experiments_A[i])
        data['InputFileNamebdot'] = 'Acc_' + str(Experiments_A[i]) + '.csv'
        data['InputFileNameTemp'] = 'Temp_const.csv'
        
    with open(path+"example.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    

    subprocess.run('python main.py example.json -n', shell=True, cwd='CFM/CFM_main')



