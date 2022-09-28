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

Experiments_Temp = ['Temp_linear']#,'Temp_const','Temp_osc','Temp_ramp']
Experiments_Acc = ['Acc_linear', 'Acc_const', 'Acc_osc','Acc_ramp']
for i in range(len(Experiments_Temp)):
    je.Terminal_run(Experiments_Temp,Experiments_Acc,i,'Temp')
    #subprocess.run('python main.py example.json -n', shell=True, cwd='CFM/CFM_main/')
#for i in range(len(Experiments_Acc)):
#    je.Terminal_run(Experiments_Temp,Experiments_Acc,i,'Acc')

'''    
saver = True
rfolder = 'CFM\CFM_main\CFMoutput_example\df'

Time,steps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,Bubble = read(rfolder,saver)
'''