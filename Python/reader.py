# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 06:15:36 2022

@author: jespe
"""

import h5py as h5
import os



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
    timesteps[0] = 0
    stps = len(timesteps)
    depth = f['depth'][1:]
    density = f['density'][:,:]
    temperature = f['temperature'][:,:]
    diffusivity = f['diffusivity'][:,1:]
    forcing = f['forcing'][:,:]
    age = f['age'][:,:]
    climate = f['Modelclimate'][:,:]
    d15N2 = f['d15N2'][:,1:]
    #print(d15N2.shape,depth.shape)
    Bubble = f['BCO'][:,1:]
    f.close()
    #with h5.File(fn,'r') as hf:
    #    print(hf.keys())
    return timesteps,stps,depth,density,temperature,diffusivity,forcing,age,climate,d15N2,Bubble
    