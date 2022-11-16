import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# -----------

data_path = '~/projects/Thesis/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'

# ice_age and corresponding depth
df3 = pd.read_excel(data_path, sheet_name='Sheet5')
ice_age_scales = np.array(df3[df3.columns[3]])
gas_age_scales = np.array(df3[df3.columns[4]])
depth_scales = np.array(df3[df3.columns[0]])

# d18O data
df = pd.read_excel(data_path, sheet_name='Sheet7')
depth_d18o = np.array(df[df.columns[1]])
d18o = np.array(df[df.columns[2]])
ice_age_d18o = np.array(df[df.columns[0]])
ice_age_d18o_err = np.array(df[df.columns[3]])

# temperature and accumulation with corresponding depth
df2 = pd.read_excel(data_path, sheet_name='Sheet4')
temp = np.array(df2[df2.columns[4]])
temp_err = np.array(df2[df2.columns[5]])
acc = np.array(df2[df2.columns[3]])
depth_temp = np.array(df2[df2.columns[0]])

# d15N data
df4 = pd.read_excel(data_path, sheet_name='Sheet6')
depth_d15n = np.array(df4[df4.columns[2]])
d15n = np.array(df4[df4.columns[3]])
d15n_err = np.array(df4[df4.columns[4]])


# ----------------------------------------------------------------------------------------------------------------------
# INTERPOLATE DATA
# ------------------

N15D = interpolate.interp1d(depth_d15n, d15n, 'linear', fill_value='extrapolate')
d15n_data_regrid = N15D(depth_scales)

N15D_err = interpolate.interp1d(depth_d15n, d15n_err, 'linear', fill_value='extrapolate')
d15n_err_regrid = N15D_err(depth_scales)

O18D = interpolate.interp1d(depth_d18o, d18o, 'linear', fill_value='extrapolate')
d18o_data_regrid = O18D(depth_scales)

AO18D = interpolate.interp1d(depth_d18o, ice_age_d18o, 'linear', fill_value='extrapolate')
ice_age_d18o_regrid = AO18D(depth_scales)

AO18D_err = interpolate.interp1d(depth_d18o, ice_age_d18o_err, 'linear', fill_value='extrapolate')
ice_age_d18o_err_regrid = AO18D_err(depth_scales)

TD = interpolate.interp1d(depth_temp, temp, 'linear', fill_value='extrapolate')
temp_regrid = TD(depth_scales)

TD_err = interpolate.interp1d(depth_temp, temp_err, 'linear', fill_value='extrapolate')
temp_err_regrid = TD_err(depth_scales)

AccD = interpolate.interp1d(depth_temp, acc, 'linear', fill_value='extrapolate')
acc_regrid = AccD(depth_scales)


# ----------------------------------------------------------------------------------------------------------------------
# WRITE TO CSV
# --------------

df_new = pd.DataFrame({"depth [m]": depth_scales, "GICC05modelext ice age [yr]": ice_age_scales,
                       "GICC05modelext gas age [yr]": gas_age_scales, "temperature [°C]": temp_regrid,
                       "temperature error [°C]": temp_err_regrid, "accumulation, tuned [m ice/yr]": acc_regrid,
                       "NGRIP d18O (permil)": d18o_data_regrid, "Age from d18O [yr]": ice_age_d18o_regrid,
                       "Counting Err Age from d18O [yr]": ice_age_d18o_err_regrid, "d15N [permil]": d15n_data_regrid,
                       "Err d15N [permil]": d15n_err_regrid})
df_new.to_excel("~/projects/Thesis/CFM_Kathi/icecore_data/data/NGRIP/interpolated_data.xlsx", index=False)

df8 = pd.read_excel(data_path2)
depth_data___ = np.flipud(np.array(df8[df8.columns[0]]))
print(depth_data___)
