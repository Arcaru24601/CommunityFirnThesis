# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:15:36 2022

@author: jespe
"""

import h5py as h5
import os
import numpy as np


def read(rfolder):
    '''
    Parameters
    ----------
    rfoler : filepath
        filepath of savefolder.
    saver : Boolean
        True for figure saving.

    Returns
    -------
    Various arrays.

    '''
    rfile = 'CFMresults.hdf5'
    fn = os.path.join(rfolder,rfile)
    f = h5.File(fn,'r')
    
    timesteps = f['DIP'][:,0]
    #timesteps[0] = 0
    stps = len(timesteps)
    depth = f['depth'][:]
    density = f['density'][:,:]
    temperature = f['temperature'][:,:]
    diffusivity = f['diffusivity'][:,:]
    forcing = f['forcing'][:,:]
    age = f['age'][:,:]
    climate = f['Modelclimate'][:,:]
    d15N2 = (f['d15N2'][:,:]-1)*1000
    d40Ar = (f[f'd40Ar'][:,:]-1)*1000
    #print(d15N2.shape,depth.shape)
    Bubble = f['BCO'][:]
    f.close()
    with h5.File(fn,'r') as hf:
        print(hf.keys())
    return timesteps,stps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,d40Ar,Bubble



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w