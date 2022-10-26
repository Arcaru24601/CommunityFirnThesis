# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:22:09 2022

@author: jespe
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
class File:
    def __init__(self,T_steps,size,Value,name,Priority,FlipFlag,sfolder='CFM/CFM_main/CFMinput/'):
        self.T_steps = T_steps
        self.Value = Value
        self.name = str(name)
        self.size = int(size)
        self.Month_sep = 6
        self.sfolder = sfolder
        self.Time_steps = np.arange(0,self.T_steps,self.Month_sep/12)
        self.Priority = Priority
        self.FlipFlag = FlipFlag
        self.RampPoint = [2000/2,2500/2]
        self.Acc_i = [0.185,0.195]

        self.Temp_i = [243,253]
        if self.Priority == 'Acc':    
            self.V_i,self.V_f = self.Acc_i
        elif self.Priority == 'Temp':    
            self.V_i,self.V_f = self.Temp_i
        self.R_i,self.R_f = self.RampPoint

        self.Folder = self.Priority + '_' + self.name
        #print(len(self.Time_steps))

    def ramp(self):
        Half_Time = np.arange(0,self.T_steps/2,self.Month_sep/12)
        slope = (self.V_f - self.V_i) / (self.R_f - self.R_i)
        y = self.V_i + np.minimum(slope * np.maximum(Half_Time - self.R_i, 0.0), self.V_f - self.V_i)
        return y
    def apply_func(self):
        if self.name == 'linear':
            #if self.Priority == 'Temp':
            Array = np.linspace(self.V_i,self.V_f,self.size)
            
            #elif self.Priority == 'Acc':
            #    Array = np.linspace(self.Acc_i[0],self.Value,len(self.Time_steps))

        elif self.name == 'const':
            Array = np.full_like(self.Time_steps,self.V_i)
        elif self.name == 'varying':
            Array = np.full_like(self.Time_steps,self.V_i)
            Array[125:250] = self.V_f
            Array[500:505] = self.V_f
            Array[1000:1100] = self.V_f
            Array[1500:2000] = self.V_f
            Array[5000:5020] = self.V_f
            Array[6000:7000] = self.V_f
            
        elif self.name == 'instant':
            A = np.full(int(self.size/2),self.V_i)
            B = np.full(int(self.size/2),self.V_f)
            Array = np.concatenate((A,B))
        elif self.name == 'osc':
            if self.Priority == 'Acc':
                F = 0.0005*2
                amplitude = 0.005
                b_0 = 0.19
                Array = amplitude * np.sin(2*np.pi*F*self.Time_steps) + b_0

            elif self.Priority == 'Temp':
                F = 0.0005*2
                amplitude = 10
                t_0 = 249
                Array = amplitude * np.sin(2*np.pi*F*self.Time_steps) + t_0
        elif self.name == 'ramp':
            t1 = 6000/2
            t0 = 4000/2
            slope = (self.V_f - self.V_i) / (t1 - t0)
            Array = self.V_i + np.minimum(slope * np.maximum(self.Time_steps - t0, 0.0), self.V_f - self.V_i)
            
        elif self.name == 'square':
            result = self.ramp()
            Array = np.concatenate((result,np.flip(result)))
            
        
        if self.Priority == 'Temp':
            Array_c = np.full_like(self.Time_steps, 0.19)
        elif self.Priority == 'Acc':
            Array_c = np.full_like(self.Time_steps, 253)
        
        if self.FlipFlag == True:
            return np.flip(Array), Array_c
        return Array,Array_c#,Folder
    def folder_gen(self):
        path = Path(self.sfolder + self.Folder)
        path.mkdir(parents=True, exist_ok=True)
    def array_gen(self):
        self.folder_gen()
        if self.Priority == 'Acc':
            Bdot_csv = np.array([self.Time_steps,self.apply_func()[0]])
            Temp_csv = np.array([self.Time_steps,self.apply_func()[1]])
            if self.FlipFlag == True:

                np.savetxt(self.sfolder + 'Flip/' + self.apply_func()[2]+'/Temp_const.csv',Temp_csv,delimiter=',')
                np.savetxt(self.sfolder + 'Flip/' + self.apply_func()[2]+'/Acc_'+self.name+'.csv',Bdot_csv,delimiter=',')
                
            np.savetxt(self.sfolder+self.Folder+'/Temp_const.csv',Temp_csv,delimiter=',')
            np.savetxt(self.sfolder+self.Folder+'/Acc_'+self.name+'.csv',Bdot_csv,delimiter=',')
                
        elif self.Priority == 'Temp':
            Temp_csv = np.array([self.Time_steps,self.apply_func()[0]])
            Bdot_csv = np.array([self.Time_steps,self.apply_func()[1]])
            if self.FlipFlag == True:

                np.savetxt(self.sfolder + 'Flip/' + self.Folder+'/Acc_const.csv',Bdot_csv,delimiter=',')
                np.savetxt(self.sfolder + 'Flip/' + self.Folder+'/Temp_'+self.name+'.csv',Temp_csv,delimiter=',')

            np.savetxt(self.sfolder+self.Folder+'/Acc_const.csv',Bdot_csv,delimiter=',')
            np.savetxt(self.sfolder+self.Folder+'/Temp_'+self.name+'.csv',Temp_csv,delimiter=',')
        
Flips = [False]
steps = 1e4/2
size = 1e4
for i in Flips:
    T_linear = File(steps,size,253,'linear','Temp',i)
    T_linear.array_gen()
    
    T_const = File(steps,size,253,'const','Temp',i)
    T_const.array_gen()
    
    T_osc = File(steps,size,253,'osc','Temp',i)
    T_osc.array_gen()


    T_ramp = File(steps,size,253,'ramp','Temp',i)
    T_ramp.array_gen()

    T_sq = File(steps,size,253,'square','Temp',i)
    T_sq.array_gen()

    T_i = File(steps,size,253,'instant','Temp',i)
    T_i.array_gen()
    
    T_v = File(steps,size,253,'varying','Temp',i)
    T_v.array_gen()
    
    
    A_linear = File(steps,size,0.22,'linear','Acc',i)
    A_linear.array_gen()

    A_const = File(steps,size,0.19,'const','Acc',i)
    A_const.array_gen()

    A_osc = File(steps,size,0.19,'osc','Acc',i)
    A_osc.array_gen()

    A_ramp = File(steps,size,0.19,'ramp','Acc',i)
    A_ramp.array_gen()

    A_sq = File(steps,size,0.19,'square','Acc',i)
    A_sq.array_gen()


    A_i = File(steps,size,0.19,'instant','Acc',i)
    A_i.array_gen()

    A_v = File(steps,size,0.19,'varying','Acc',i)
    A_v.array_gen()



