# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:23:01 2022

@author: jespe
"""


#from reader import read
import seaborn as sns 
sns.set()
import os,subprocess,sys,json
import Test_script as je
from pathlib import Path
folder = './CFM/CFM_main/CFMinput'

Folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
Advection = ['zero','Christo','Darcy']
Effect = ['grav','full']
def run(folder):
    for i in range(len(folder)):
        path = Path('CFM/CFM_main/CFMoutput/' + folder[i])
        path.mkdir(parents=True, exist_ok=True)
        print(Folder[i])
        if folder[i].startswith('Temp'):
            je.Terminal_run(folder,i,'Temp')
        elif folder[i].startswith('Acc'):
            je.Terminal_run(folder,i,'Acc')
folder2 = './CFM/CFM_main/CFMinput/DO_event'

Folder2 = [name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))]

def Air_run(Model,folder,advec,effect):
    for i in range(len(folder)):
        for j in range(len(advec)):
            
            for k in range(len(effect)):
                Foldname = str(folder[i]) + '/' + str(Model) + '/' + str(advec[j]) + '/' + str(effect[k])
                path = Path('CFM/CFM_main/CFMoutput/DO_event/' + Foldname)
                path.mkdir(parents=True, exist_ok=True)
                print(Model,folder[i],advec[j],effect[k])
                je.Terminal_run2(Model,folder[i],'Temp',effect[k],advec[j])


#run(Folder)                
#Air_run('Barnola1991',Folder2,Advection,Effect)
#Models = ['HLdynamic','Barnola1991','Goujon2003']
#from multiprocessing import Process
'''
if __name__ == "__main__":
    processes = [Process(target = Air_run, args = ('HLdynamic',  Folder2, Advection,Effect)),
                 Process(target = Air_run, args = ('Barnola1991',Folder2, Advection,Effect))]
                 Process(target = Air_run, args = ('Goujon2003', Folder2, Advection,Effect)),
                 
    # kick them off 
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()


'''
folder2 = './CFM/CFM_main/CFMinput/Equi'

#Equi_Folder = [name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))]
Exp = ['Temp','Acc','Both']
Models = ['HLdynamic','Barnola1991','Goujon2003']
Folder = ['50y','200y','500y','1000y','2000y']
Folder_Amp = ['0.3','0.5','1.0','2.0','3.0']
def Equi_run(Model,exp,folder):
    for k in range(len(exp)):
        for j in range(len(Model)):
        
            for i in range(len(folder)):
            
                print(exp[k],Model[j],folder[i])
                je.Terminal_run_Models(Model[j],exp[k],folder[i])
                



Equi_run(Models,Exp,Folder)


def Equi_run_Amp(Model,exp,folder):
    for k in range(len(exp)):
        for j in range(len(Model)):
        
            for i in range(len(folder)):
            
                print(exp[k],Model[j],folder[i])
                je.Terminal_run_Amp(Model[j], exp[k], folder[i])
Equi_run_Amp(Models,Exp,Folder_Amp)



