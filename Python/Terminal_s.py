# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:46:42 2022

@author: jespe
"""

import json,subprocess



def Terminal_run(Experiments_T,Experiments_A,i,priority):
    file = open('CFM/CFM_main/example.json')
        
    data = json.load(file)
    if priority == 'Temp':
        data['resultsfolder'] = 'CFMoutput/' + str(Experiments_T[i])
        data['InputFileFolder'] = 'CFMinput/' + str(Experiments_T[i])
        data['InputFileNameTemp'] = 'Temp_' + str(Experiments_T[i]) + '.csv'
        data['InputFileNamebdot'] = 'Acc_const.csv'
        
    elif priority == 'Acc':
        data['resultsFolder'] = 'CFMoutput/' + str(Experiments_A[i])
        data['InputFileFolder'] = 'CFMinput/' + str(Experiments_A[i])
        data['InputFileNamebdot'] = 'Acc_' + str(Experiments_A[i]) + '.csv'
        data['InputFileNameTemp'] = 'Temp_const.csv'
        
    with open("CFM/CFM_main/example.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    

    subprocess.run('python main.py example.json -n', shell=True, cwd='CFM/CFM_main/')




