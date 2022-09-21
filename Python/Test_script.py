# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:57:34 2022

@author: jespe
"""

import json,os
import numpy as np

path = 'CFM/CFM_main/'
file = open(path+'example.json')

data = json.load(file)
data['int_type'] = 'linear'

with open(path+"example.json", 'w') as f:
    json.dump(data, f,indent = 2)
    
# Closing file
f.close()