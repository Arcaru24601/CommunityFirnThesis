# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:09:39 2023

@author: jespe
"""

from csaps import CubicSmoothingSpline
import numpy as np

def smooth_param(cop, x): #cop cut-off period
    dx = np.mean(np.diff(x)) # mean dist betw points
    lamda = (1/ (2 * cop * np.pi))**4 / dx
    p = 1 / (1 + lamda)
    return p

def smooth_data(cop, ydata2smooth, xdata, new_grid):
    p = smooth_param(cop, xdata)
    sp = CubicSmoothingSpline(xdata, ydata2smooth, smooth = p)
    y_smooth = sp(new_grid)
    return y_smooth, new_grid

def rem_neg_val(data):
    if np.any(np.diff(data) < 0):
        ind = np.where(np.diff(data) < 0)
        data[ind] = data[ind + 1]
        print('Value changed at %5.3f to %5.3f', (ind, ind +1))
    return data
