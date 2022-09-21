# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:22:09 2022

@author: jespe
"""

import csv
import numpy as np
path = 'CFM/CFM_main/CFMinput_example/'



class File:
    def __init__(self,T_steps,size,Value,name,Acc_const,T_const):
        self.T_steps = T_steps
        self.Value = Value
        self.name = str(name)
        self.Acc_const = Acc_const
        self.T_const = T_const
        self.size = int(size)
        
    def apply_func(self):
        if self.name == 'linear':
            if self.T_const == None:
                Array = np.linspace(243,self.Value,self.size)
                Folder = 'Temp_linear'
            elif self.Acc_const == None:
                Array = np.linspace(0.12,self.Value,self.size)
                Folder = 'Acc_linear'
        elif self.name == 'const':
            Array = np.full(self.size,self.Value)
            if self.Acc_const == None:
                Folder = 'Acc_const'
            elif self.T_const == None:
                Folder = 'T_const'
        elif self.name == 'osc':
            if self.Acc_const == None:
                F = 0.001
                T = 10000
                amplitude = 0.1
                t = np.arange(0, T, 1)
                b_0 = 0.1
                Array = amplitude * np.sin(2*np.pi*F*t) + b_0
                Folder = 'Acc_osc'
            elif self.T_const == None:
                F = 0.001
                T = 10000
                amplitude = 20
                t_0 = 240
                t = np.arange(0, T, 1)
                Array = amplitude * np.sin(2*np.pi*F*t) + t_0
                Folder = 'T_osc'
        return Array,Folder
    def array_gen(self):
        Time_steps = np.arange(0,self.T_steps,self.T_steps/self.size)
        if self.Acc_const == None:
            Bdot_csv = np.array([Time_steps,self.apply_func()[0]])
            Temp_csv = np.array([Time_steps,np.full_like(Time_steps,self.T_const)])
            np.savetxt(self.apply_func()[1]+'/Temp_const.csv',Temp_csv,delimiter=',')
            np.savetxt(self.apply_func()[1]+'/Acc_'+self.name+'.csv',Bdot_csv,delimiter=',')

        elif self.T_const == None:
            Temp_csv = np.array([Time_steps,self.apply_func()[0]])
            Bdot_csv = np.array([Time_steps,np.full_like(Time_steps,self.Acc_const)])
            np.savetxt(self.apply_func()[1]+'/Acc_const.csv',Bdot_csv,delimiter=',')
            np.savetxt(self.apply_func()[1]+'/Temp_'+self.name+'.csv',Temp_csv,delimiter=',')
        
            


T_linear = File(1e4,1e4,253,'linear',0.19,None)
T_linear.array_gen()

T_const = File(1e4,1e4,253,'const',0.19,None)
T_const.array_gen()

T_osc = File(1e4,1e4,253,'osc',0.19,None)
T_osc.array_gen()



A_linear = File(1e4,1e4,0.22,'linear',None,245)
A_linear.array_gen()

A_const = File(1e4,1e4,0.19,'const',None,245)
A_const.array_gen()

A_osc = File(1e4,1e4,0.19,'osc',None,245)
A_osc.array_gen()


