# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:39:00 2022

@author: jespe
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:22:09 2022

@author: jespe
"""

import csv
import numpy as np
path = 'CFM/CFM_main/CFMinput_example/'


class File:
    def __init__(self,T_steps,size,Value,name,Acc_const,T_const,sfolder='CFM/CFM_main/CFMinput/'):
        self.T_steps = T_steps
        self.Value = Value
        self.name = str(name)
        self.Acc_const = Acc_const
        self.T_const = T_const
        self.size = int(size)
        self.sfolder = str(sfolder)
        self.Time_steps = np.arange(0,self.T_steps,self.T_steps/self.size)

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
                Folder = 'Temp_const'
        elif self.name == 'osc':
            if self.Acc_const == None:
                F = 0.0005
                amplitude = 0.005
                t = np.arange(0,self.T_steps,self.T_steps/self.size)
                b_0 = 0.19
                Array = amplitude * np.sin(2*np.pi*F*t) + b_0
                Folder = 'Acc_osc'

            elif self.T_const == None:
                F = 0.0005
                amplitude = 10
                t_0 = 249
                t = np.arange(0,self.T_steps,self.T_steps/self.size)
                Array = amplitude * np.sin(2*np.pi*F*t) + t_0
                Folder = 'Temp_osc'
        elif self.name == 'ramp':
            if self.Acc_const == None:
                t_f = 0.195
                t_i = 0.185
                t1 = 6000
                t0 = 4000
                slope = (t_f - t_i) / (t1 - t0)
                Folder = 'Acc_ramp'
            elif self.T_const == None:
                t_f = 243.5
                t_i = 233.5
                t1 = 6000
                t0 = 4000
                slope = (t_f - t_i) / (t1 - t0)
                Folder = 'Temp_ramp'
            Array = t_i + np.minimum(slope * np.maximum(self.Time_steps - t0, 0.0), t_f - t_i)
            print(Array.max(),Array.min())
        return Array,Folder
    def array_gen(self):
        if self.Acc_const == None:
            Bdot_csv = np.array([self.Time_steps,self.apply_func()[0]])
            Temp_csv = np.array([self.Time_steps,np.full_like(self.Time_steps,self.T_const)])
            np.savetxt(self.sfolder+self.apply_func()[1]+'/Temp_const.csv',Temp_csv,delimiter=',')
            np.savetxt(self.sfolder+self.apply_func()[1]+'/Acc_'+self.name+'.csv',Bdot_csv,delimiter=',')

        elif self.T_const == None:
            Temp_csv = np.array([self.Time_steps,self.apply_func()[0]])
            Bdot_csv = np.array([self.Time_steps,np.full_like(self.Time_steps,self.Acc_const)])
            np.savetxt(self.sfolder+self.apply_func()[1]+'/Acc_const.csv',Bdot_csv,delimiter=',')
            np.savetxt(self.sfolder+self.apply_func()[1]+'/Temp_'+self.name+'.csv',Temp_csv,delimiter=',')
        

steps = 1e4
size = 1e4

T_linear = File(steps,size,253,'linear',0.19,None)
T_linear.array_gen()

T_const = File(steps,size,253,'const',0.19,None)
T_const.array_gen()

T_osc = File(steps,size,253,'osc',0.19,None)
T_osc.array_gen()


T_ramp = File(steps,size,253,'ramp',0.19,None)
T_ramp.array_gen()



A_linear = File(steps,size,0.22,'linear',None,245)
A_linear.array_gen()

A_const = File(steps,size,0.19,'const',None,245)
A_const.array_gen()

A_osc = File(steps,size,0.19,'osc',None,245)
A_osc.array_gen()

A_ramp = File(steps,size,0.19,'ramp',None,237.5)
A_ramp.array_gen()





class Csv_gen(File):
    def __init__(self,T_steps, size, Value, name, Acc_const, T_const,Priority,FlipFlag):
        super().__init__(T_steps, size, Value, name, Acc_const, T_const)
        self.Priority = Priority
        self.FlipFlag = FlipFlag
    def Csv_generation(self):
        print(self.apply_func())




Test = Csv_gen(steps, size, 253, 'linear', 0.19, None, None, None)
Test.Csv_generation()








