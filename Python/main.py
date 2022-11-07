# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:23:01 2022

@author: jespe
"""


from reader import read
import seaborn as sns 
sns.set()
import os,subprocess,sys,json
import Test_script as je
from pathlib import Path
folder = './CFM/CFM_main/CFMinput'

Folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]


Experiments_Temp = ['Temp_linear','Temp_const','Temp_osc','Temp_ramp','Temp_square','Temp_instant','Temp_varying']
Experiments_Acc = ['Acc_linear', 'Acc_const', 'Acc_osc','Acc_ramp','Acc_square','Acc_instant','Acc_varying']
FlipFlags = [False]
'''
for j in range(len(FlipFlags)):
    for i in range(len(Experiments_Temp)):
        path = Path('CFM/CFM_main/CFMoutput/' + Experiments_Temp[i])
        path.mkdir(parents=True, exist_ok=True)
        je.Terminal_run(Experiments_Temp,Experiments_Acc,i,'Temp',FlipFlags[j])
    #subprocess.run('python main.py example.json -n', shell=True, cwd='CFM/CFM_main/')
    for i in range(len(Experiments_Acc)):
        path = Path('CFM/CFM_main/CFMoutput/' + Experiments_Acc[i])
        path.mkdir(parents=True, exist_ok=True)
        je.Terminal_run(Experiments_Temp,Experiments_Acc,i,'Acc',FlipFlags[j])
'''

for i in range(len(Folder)):
    path = Path('CFM/CFM_main/CFMoutput/' + Folder[i])
    path.mkdir(parents=True, exist_ok=True)
    if Folder[i].startswith('Temp'):
        je.Terminal_run(Folder,i,'Temp')
    elif Folder[i].startswith('Acc'):
        je.Terminal_run(Folder,i,'Acc')


'''
saver = True
rfolder = 'CFM\CFM_main\CFMoutput_example\df'

Time,steps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,Bubble = read(rfolder,saver)
'''