# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:41:12 2023

@author: jespe
"""

import h5py as hf
import numpy as np
from matplotlib import pyplot as plt


def get_T_surf(file):
    T_surf = file['temperature'][:, 1]
    return T_surf


def get_cod(file, mode):
    if mode == 'CoD':  # Martinerie close-off depth
        cod = file['BCO'][:, 2]
        

    elif mode == 'LiD':
        cod = file['BCO'][:, 6]

    elif mode == 'zero_diff':
        diffusivity = file['diffusivity'][:]
        depth = file['depth']
        cod = np.zeros(np.shape(diffusivity)[0])
        index = np.zeros(np.shape(diffusivity)[0])
        for i in range((np.shape(diffusivity)[0])):
            index[i] = np.max(np.where(diffusivity[i, 1:] > 10**(-20))) + 1
            cod[i] = depth[i, int(index[i])]
    return cod


def get_T_cod(file, mode):
    depth = file['depth'][:]
    
    cod = get_cod(file, mode)
    T = file['temperature'][:]
    T_cod = np.ones_like(cod)
    
    for i in range(depth.shape[0]):
        #print(np.where(depth[i, 1:] == cod[i]))
        ind = int(np.where(depth[i, 1:] == cod[i])[0])
        T_cod[i] = T[i, ind]
    return T_cod


def get_T_mean(file, mode):
    T_surf = get_T_surf(file)
    T_cod = get_T_cod(file, mode)
    
    T_means = np.zeros_like(T_cod)
    for i in range(np.shape(T_surf)[0]):
        if T_surf[i] > T_cod[i]:
            T_means[i] = (T_cod[i] * T_surf[i]) / (T_surf[i] -
                                                   T_cod[i]) * np.log(T_surf[i] / T_cod[i])
        else:
            T_means[i] = (T_cod[i] * T_surf[i]) / (T_cod[i] -
                                                   T_surf[i]) * np.log(T_cod[i] / T_surf[i])

    return T_means


def get_d15N_therm(file, mode):  # FInd formulation for ALpha for argon
    T_means = get_T_mean(file, mode)
    T_surf = get_T_surf(file)
    T_cod = get_T_cod(file, mode)
    alpha = (8.656 - (1232/T_means))*10**(-3)
    
    d15N_therm = ((T_surf / T_cod)**alpha - 1)*1000
    return d15N_therm()


def get_d15N_grav(file, mode):
    #T_means = get_T_mean(file, mode)
    T_means = get_T_cod(file, mode)
    cod = get_cod(file, mode)
    Grav = 9.81
    R = 8.3145
    delta_M = 1/1000  # Why is this 1/1000
    #print(T_means)
    d15N_grav = (np.exp((delta_M*Grav*cod) / (R*T_means)) - 1) * 1000
    return d15N_grav


def get_d15N_tot(file, mode):
    d15N_therm = get_d15N_therm(file, mode)
    d15N_grav = get_d15N_grav(file, mode)
    return d15N_therm + d15N_grav
