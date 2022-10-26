# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:23:17 2022

@author: jespe
"""

import os

folder = './CFM/CFM_main/CFMinput'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

print(sub_folders)