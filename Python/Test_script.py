# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:46:42 2022

@author: jespe
"""

import json,subprocess



def Terminal_run(Folder,i,priority,FlipFlag=False):
    file = open('CFM/CFM_main/example.json')
        
    data = json.load(file)
    data['grid_outputs'] = False
    data['resultsFileName'] = 'CFMresults.hdf5'
    data['resultsFolder'] = 'CFMoutput/' + str(Folder[i])
    data['InputFileFolder'] = 'CFMinput/' + str(Folder[i])
    if priority == 'Temp':
        data['InputFileNameTemp'] = str(Folder[i])+'.csv'
        data['InputFileNamebdot'] = 'Acc_const.csv'
        if FlipFlag == True:
            data['InputFileFolder'] = 'CFMinput/Flip/' + str(Folder[i])
            data['resultsFolder'] = 'CFMoutput/Flip/' + str(Folder[i])
        
    elif priority == 'Acc':
        data['InputFileNamebdot'] = str(Folder[i])+'.csv'
        data['InputFileNameTemp'] = 'Temp_const.csv'  
        if FlipFlag == True:
            data['InputFileFolder'] = 'CFMinput/Flip/' + str(Folder[i])
            data['resultsFolder'] = 'CFMoutput/Flip/' + str(Folder[i])
        
    with open("CFM/CFM_main/example.json", 'w') as f:
        json.dump(data, f,indent = 2)
        
        # Closing file
    f.close()    

    subprocess.run('python main.py example.json -n', shell=True, cwd='CFM/CFM_main/')




