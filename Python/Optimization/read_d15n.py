# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:53:49 2023

@author: jespe
"""

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from Cubic_smooth import *
import h5py as hf
import matplotlib.pyplot as plt
from d18O_read import get_interval_data_NoTimeGrid, find_start_end_ind
from calc_d15N import *

def get_d15N_data(path_data):
    
    df = pd.read_excel(path_data, sheet_name='Sheet6')
    depth_d15n = np.flipud(np.array(df[df.columns[2]]))
    d15n = np.flipud(np.array(df[df.columns[3]]))
    d15n_err = np.flipud(np.array(df[df.columns[4]]))
    
    # Ice age depths
    # ice age depths
    df2 = pd.read_excel(path_data, sheet_name='Sheet5')
    ice_age_scales = np.flipud(np.array(df2[df2.columns[3]])) * (-1)
    gas_age_scales = np.flipud(np.array(df2[df2.columns[4]])) * (-1)
    depth_scales = np.array(df2[df2.columns[0]])
    
    #Interpolate
    IAD = interpolate.interp1d(depth_scales, ice_age_scales, 'linear', fill_value='extrapolate')                           
    print(depth_d15n)
    ice_age_d15n = IAD(depth_d15n)
    
    GAD = interpolate.interp1d(depth_scales, gas_age_scales, 'linear', fill_value='extrapolate')                           
    gas_age_d15n = GAD(depth_d15n)
    
    return ice_age_d15n, gas_age_d15n, d15n, d15n_err

def get_d15N_data_interval(path_data, ice_age_d15N_model):
    ice_age_d15n, gas_age_d15n, d15n, d15n_err = get_d15N_data(path_data)
    start_year = np.min(ice_age_d15N_model)
    end_year = np.max(ice_age_d15N_model)
    
    start_ind, end_ind = find_start_end_ind(ice_age_d15n, start_year, end_year)
    ice_age_d15n_interval = ice_age_d15n[start_ind: end_ind]
    gas_age_d15n_interval = gas_age_d15n[start_ind: end_ind]
    d15n_interval = d15n[start_ind: end_ind]
    d15n_err_interval = d15n_err[start_ind: end_ind]
    
    
    return ice_age_d15n_interval, gas_age_d15n_interval, d15n_interval, d15n_err_interval

def interp_d15Nmodel_to_d15Ndata(d15n_model, ice_age_model, gas_age_model, ice_age_data):
    Data_model = interpolate.interp1d(ice_age_model, d15n_model, 'linear', fill_value='extrapolate')
    d15n_model_interp_ice_age = Data_model(ice_age_data)
    
    IA_GA = interpolate.interp1d(ice_age_model, gas_age_model, 'linear', fill_value='extrapolate')
    gas_age_model_interp = IA_GA(ice_age_data)
    return d15n_model_interp_ice_age, gas_age_model_interp




def get_d15N_model(path_model, mode, FirnAir, cop):
    file = hf.File(path_model, 'r')
    depth_model = file['depth'][:]
    ice_age_model = file['age'][:]
    
    if FirnAir:
        if mode == 'CoD':
            close_off_depth = file['BCO'][:,2]
            gas_age_model = file['gas_age'][:]
            d15n_model = file['d15N2'][:] -1
            d15n_cod = np.ones_like(close_off_depth)
            gas_age_cod = np.ones_like(close_off_depth)
            ice_age_cod = np.ones_like(close_off_depth)
            
        elif mode == 'LiD':
            close_off_depth = file['BCO'][:,6]
            gas_age_model = file['gas_age'][:]
            d15n_model = file['d15N2'][:] -1
            d15n_cod = np.ones_like(close_off_depth)
            gas_age_cod = np.ones_like(close_off_depth)
            ice_age_cod = np.ones_like(close_off_depth)
        
        elif mode == 'zero_diff':
            diffusivity = file['diffusivity'][:]
            gas_age_model = file['gas_age'][:]
            d15n_model = file['d15N2'][:] -1
            index = np.zeros(np.shape(diffusivity)[0])
            ice_age_cod = np.zeros(np.shape(diffusivity)[0])
            gas_age_cod = np.zeros(np.shape(diffusivity)[0])
            d15n_cod = np.zeros(np.shape(diffusivity)[0])
            close_off_depth = np.zeros(np.shape(diffusivity)[0])
            
            for i in range(np.shape(diffusivity)[0]):
                index[i] = np.max(np.where(diffusivity[i, 1:] > 10**(-20))) +1
                ice_age_cod[i] = ice_age_model(i, int(index[i]))
                gas_age_cod[i] = gas_age_model(i, int(index[i]))

                d15n_cod[i] = d15n_model(i, int(index[i]))
                close_off_depth[i] = depth_model(i, int(index[i]))
            
        for i in range(depth_model.shape[0]):
            index = int(np.where(depth_model[i, 1:] == close_off_depth[i])[0])
            gas_age_cod[i] = gas_age_model[i, index]
            ice_age_cod[i] = ice_age_model[i, index]
            d15n_cod[i] = d15n_model[i, index]
            
        gas_age_cod = gas_age_cod[1:]
        ice_age_cod = ice_age_cod[1:]
        d15n_cod = d15n_cod[1:] * 1000
        modeltime = depth_model[1:, 0]
        
        gas_age_cod_smooth = smooth_data(cop, gas_age_cod, modeltime, modeltime)[0]
        ice_age_cod_smooth = smooth_data(cop, ice_age_cod, modeltime, modeltime)[0]
        
        ice_age = modeltime - ice_age_cod_smooth # Weird stuff modeltime
        delta_age = ice_age_cod - gas_age_cod_smooth
        gas_age = ice_age + delta_age
        
    else:
        d15n_cod = d15N_tot(file = file, mode=mode)
        close_off_depth = get_cod(file=file, mode=mode)
        ice_age_cod = np.ones_like(close_off_depth)
        
        for i in range(depth_model.shape[0]):
            index = int(np.where(depth_model[i, 1:] == close_off_depth[i])[0])
            ice_age_cod[i] = ice_age_model[i, index]
        ice_age_cod = ice_age_cod[1:]
        
        gas_age_cod = np.zeros_like(ice_age_cod)
        modeltime = depth_model[1:, 0]
        ice_age_cod_smooth = smooth_data(cop, ice_age_cod, modeltime, modeltime)[0]
        
        ice_age = modeltime - ice_age_cod_smooth
        delta_age = ice_age_cod_smooth - gas_age_cod
        gas_age = ice_age + delta_age
        d15n_cod = d15n_cod[1:]
        
    return d15n_cod, ice_age, gas_age, delta_age

            
            
            
            
            
Test = get_d15N_data('data/NGRIP/supplement.xlsx')            


    
    
