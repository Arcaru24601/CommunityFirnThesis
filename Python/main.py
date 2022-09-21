# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:23:01 2022

@author: jespe
"""

from plot import plotter
from reader import read
from Terminal_script import Terminal_run

from reader import read
import seaborn as sns 
sns.set()



Experiments_Temp = ['Temp_linear']#,'Temp_const','osc_Temp']
Experiments_Acc = ['Acc_linear']#, 'Acc_const', 'osc_Acc']
for i in range(len(Experiments_Temp)):
    Terminal_run(Experiments_Temp,Experiments_Acc,i,'Temp')
for i in range(len(Experiments_Acc)):
    Terminal_run(Experiments_Temp,Experiments_Acc,i,'Acc')

    
saver = True
rfolder = 'CFM\CFM_main\CFMoutput_example\df'

Time,steps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,Bubble = read(rfolder,saver)
