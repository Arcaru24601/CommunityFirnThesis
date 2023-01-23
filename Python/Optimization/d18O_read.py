# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:35:20 2023

@author: jespe
"""

import numpy as np
import pandas as pd
from Cubic_smooth import smooth_data

def read_data_d18O(path_data):
    df = pd.read_excel(path_data)
    depth_data = np.flipud(np.array(df[df.columns[0]]))
    d18O_data = np.flipud(np.array(df[df.columns[6]])) / 1000.
    ice_age_data = np.flipud(np.array(df[df.columns[7]])) * (-1)
    return depth_data, d18O_data, ice_age_data


def find_start_end_ind(ice_age_data, start_year, end_year):
    t_start_ind = np.min(np.where(ice_age_data >= start_year))
    t_end_ind = np.max(np.where(ice_age_data <= end_year))
    return t_start_ind, t_end_ind



def get_interval_data(depth_data, d18O_data, ice_age_data, start_year, end_year, dt, cop):
    t_start_ind, t_end_ind = find_start_end_ind(ice_age_data, start_year, end_year)
    depth_data_interval = depth_data[t_start_ind: t_end_ind]
    d18O_interval = d18O_data[t_start_ind: t_end_ind]
    ice_age_interval = ice_age_data[t_start_ind: t_end_ind]
    time_grid = new_time_grid(start_year, end_year, dt)
    d18O_smooth, time_grid = smooth_data(
        cop, d18O_interval, ice_age_interval, time_grid)
    return depth_data_interval, d18O_interval, ice_age_interval, d18O_smooth, time_grid


def get_interval_data_NoTimeGrid(depth_data, d18O_data, ice_age_data, start_year, end_year):
    t_start_ind, t_end_ind = find_start_end_ind(ice_age_data, start_year, end_year)
    depth_data_interval = depth_data[t_start_ind: t_end_ind]
    d18O_interval = d18O_data[t_start_ind: t_end_ind]
    ice_age_interval = ice_age_data[t_start_ind: t_end_ind]

    return depth_data_interval, d18O_interval, ice_age_interval


def new_time_grid(T_init, T_final, dt):
    time_grid = np.arange(T_init, T_final, dt)
    return time_grid
