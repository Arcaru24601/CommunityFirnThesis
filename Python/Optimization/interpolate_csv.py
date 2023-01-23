# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:42:09 2023

@author: jespe
"""

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from matplotlib import pyplot as plt


# =============================================================================
# Data loading
# =============================================================================

path = 'data/NGRIP/supplement.xlsx'

# ice age depths
df = pd.read_excel(path, sheet_name='Sheet5')
ice_age_scale = np.array(df[df.columns[3]])
gas_age_scale = np.array(df[df.columns[4]])
depth_scale = np.array(df[df.columns[0]])

# d18O data
df2 = pd.read_excel(path, sheet_name='Sheet7')
depth_d18o = np.array(df2[df2.columns[1]])
d18o = np.array(df2[df2.columns[2]])
ice_age_d18o = np.array(df2[df2.columns[0]])
ice_age_d18o_err = np.array(df2[df2.columns[3]])



# temp, acc and depths
df3 = pd.read_excel(path, sheet_name='Sheet4')
temp = np.array(df3[df3.columns[4]])
temp_err = np.array(df3[df3.columns[5]])
acc = np.array(df3[df3.columns[3]])
temp_depth = np.array(df3[df3.columns[0]])



# temp, acc and depths
df4 = pd.read_excel(path, sheet_name='Sheet6')
depth_d15n = np.array(df4[df4.columns[2]])
d15n = np.array(df4[df4.columns[3]])
d15n_err = np.array(df4[df4.columns[4]])






# =============================================================================
# Interpolate Data
# =============================================================================

N15D = interpolate.interp1d(depth_d15n, d15n, 'linear', fill_value = 'extrapolate')
d15n_regrid = N15D(depth_scale)


N15D_err = interpolate.interp1d(depth_d15n, d15n_err, 'linear', fill_value = 'extrapolate')
d15n_err_regrid = N15D_err(depth_scale)


O18D = interpolate.interp1d(depth_d18o, d18o, 'linear', fill_value = 'extrapolate')
d18o_regrid = O18D(depth_scale)


AO18D = interpolate.interp1d(depth_d18o, ice_age_d18o, 'linear', fill_value = 'extrapolate')
ice_age_d18o_regrid = AO18D(depth_scale)


AO18D_err = interpolate.interp1d(depth_d18o, ice_age_d18o_err, 'linear', fill_value = 'extrapolate')
ice_age_d18o_err_regrid = AO18D_err(depth_scale)

TD = interpolate.interp1d(temp_depth, temp, 'linear', fill_value = 'extrapolate')
temp_regrid = TD(depth_scale)


TD_err = interpolate.interp1d(temp_depth, temp_err, 'linear', fill_value = 'extrapolate')
temp_err_regrid = TD_err(depth_scale)


AccD = interpolate.interp1d(temp_depth, acc, 'linear', fill_value = 'extrapolate')
acc_regrid = AccD(depth_scale)


# =============================================================================
# Write to csv and excel
# =============================================================================




df_new = pd.DataFrame({"depth [m]": depth_scale, "GICC05modelext ice age [yr]": ice_age_scale,
                       "GICC05modelext gas age [yr]": gas_age_scale, "temperature [°C]": temp_regrid,
                       "temperature error [°C]": temp_err_regrid, "accumulation, tuned [m ice/yr]": acc_regrid,
                       "NGRIP d18O (permil)": d18o_regrid, "Age from d18O [yr]": ice_age_d18o_regrid,
                       "Counting Err Age from d18O [yr]": ice_age_d18o_err_regrid, "d15N [permil]": d15n_regrid,
                       "Err d15N [permil]": d15n_err_regrid})


df_new.to_excel('data/NGRIP/interpolated.xlsx', index = False)


