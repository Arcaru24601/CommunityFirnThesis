# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:27:02 2023

@author: jespe
"""

import pandas as np
import numpy as np
from d18O_read import *
from matplotlib import pyplot as plt


def read_temp(path_data):
    df = pd.read_excel(path_data)
    temp = np.flipud(np.array(df[df.columns[3]]))
    temp_err = np.flipud(np.array(df[df.columns[4]]))
    return temp, temp_err


def read_acc(path_data):
    df = pd.read_excel(path_data)
    acc = np.flipud(np.array(df[df.columns[5]]))

    return acc


def get_interval_temp(temp, temp_err, ice_age_data, start_year, end_year):
    t_start_ind, t_end_ind = find_start_end_ind(ice_age_data, start_year, end_year)
    
    temp_interval = temp[t_start_ind: t_end_ind]
    temp_err_interval = temp_err[t_start_ind: t_end_ind]
    return temp_interval, temp_err_interval


def get_interval_acc(acc, ice_age_data, start_year, end_year):
    t_start_ind, t_end_ind = find_start_end_ind(ice_age_data, start_year, end_year)
    #print(t_start_ind,t_end_ind)
    acc_interval = acc[t_start_ind: t_end_ind]
    return acc_interval


if __name__ == '__main__':
    data_path = 'data/NGRIP/interpolated.xlsx'
    start_year = -120000
    end_year_ = -10000
    depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
    temp, temp_err = read_temp(data_path)
    acc = read_acc(data_path)
    depth_interval, d18O_interval, ice_age_interval = get_interval_data_NoTimeGrid(depth_full, d18O_full,ice_age_full,start_year=-120000, end_year=-10000)

    acc_interval = get_interval_acc(acc, ice_age_full, start_year=-120000, end_year=-10000)

    input_acc = np.array([ice_age_interval, acc_interval])
    plt.plot(input_acc[0], input_acc[1])
    plt.show()
