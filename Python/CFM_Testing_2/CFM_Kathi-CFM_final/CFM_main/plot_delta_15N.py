import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Data from model ------------------------------------------------------------------------------------------------------
file = h5py.File('results/NGRIP_Goujon/CFMresults_NGRIP_Goujon.hdf5', 'r')
d15n_model = file["d15N2"][-1, 1:] - 1.
gas_age_model = file["age"][-1, 1:]
depth_model = file['depth']

# lower and upper temperature bounds
file2 = h5py.File('results/NGRIP_Goujon_up/CFMresults_NGRIP_up_Gou.hdf5', 'r')
d15n_model_up = file2["d15N2"][-1, 1:] - 1.
gas_age_model2 = file2["age"][-1, 1:]
depth_model2 = file2['depth']

file3 = h5py.File('results/NGRIP_Goujon_lo/CFMresults_NGRIP_lo_Gou.hdf5', 'r')
d15n_model_lo = file3["d15N2"][-1, 1:] - 1.
gas_age_model3 = file3["age"][-1, 1:]
depth_model3 = file3['depth']





# Data from ice core ---------------------------------------------------------------------------------------------------
file_location = '~/projects/Thesis/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'
df3 = pd.read_excel(file_location, sheet_name='Sheet6')
depth_data = np.array(df3[df3.columns[2]])
d15n_data = np.array(df3[df3.columns[3]])
d15n_data_err = np.array(df3[df3.columns[4]])
d15n_data_up = d15n_data + d15n_data_err
d15n_data_lo = d15n_data - d15n_data_err

df2 = pd.read_excel(file_location, sheet_name='Sheet5')
gas_age = np.array(df2[df2.columns[3]])
depth = np.array(df2[df2.columns[0]])

# get depth from data corresponding to the modelled gas_age --> depth_regrid
MD = interpolate.interp1d(gas_age, depth, 'linear', fill_value='extrapolate')
gas_age_model_regrid = gas_age_model + gas_age[0]
depth_regrid = MD(gas_age_model_regrid)

# for upper and lower temperature bounds
gas_age_model2_regrid = gas_age_model2 + gas_age[0]
gas_age_model3_regrid = gas_age_model3 + gas_age[0]

# get corresponding d15N values
ND = interpolate.interp1d(depth_data, d15n_data, 'linear', fill_value='extrapolate')
d15n_data_regrid = ND(depth_regrid)
ND_up = interpolate.interp1d(depth_data, d15n_data_up, 'linear', fill_value='extrapolate')
d15n_data_up_regrid = ND_up(depth_regrid)
ND_lo = interpolate.interp1d(depth_data, d15n_data_lo, 'linear', fill_value='extrapolate')
d15n_data_lo_regrid = ND_lo(depth_regrid)


# this is the shifted version (shift by 1100 yr)
gas_age_model_regrid2 = gas_age_model + gas_age[0] + 1100
depth_regrid2 = MD(gas_age_model_regrid2)
d15n_data_regrid2 = ND(depth_regrid2)
d15n_data_up_regrid2 = ND_up(depth_regrid2)
d15n_data_lo_regrid2 = ND_lo(depth_regrid2)



# Plot -----------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, sharex=False, sharey=False)
fig.set_figheight(7)
fig.set_figwidth(10)
# fig.suptitle('Sharing both axes')
axs[0].plot(gas_age_model_regrid, d15n_data_regrid, 'k--', label='$\delta^{15}$N$_2$ data', linewidth=0.5)
axs[0].fill_between(gas_age_model_regrid, d15n_data_lo_regrid, d15n_data_up_regrid, alpha=0.2, facecolor='k')
axs[0].plot(gas_age_model_regrid, d15n_model * 1000, 'r-', label='Goujon', linewidth=0.9)
axs[0].plot(gas_age_model2_regrid, d15n_model_up * 1000, 'r-.', label='Goujon $\Delta$T = +3K', linewidth=0.9, alpha=0.5)
axs[0].plot(gas_age_model3_regrid, d15n_model_lo * 1000, 'r:', label='Goujon $\Delta$T = -3K', linewidth=0.9, alpha=0.5)
axs[0].fill_between(gas_age_model_regrid, d15n_model_lo * 1000, d15n_model_up * 1000, alpha=0.2, facecolor='r')
axs[0].grid(linestyle='--', color='gray', lw='0.5')
axs[0].set_ylabel(r"$\delta^{15}$N$_2$ [‰]")
axs[0].set_xlabel(r"gas age GICC05modelext [yr]")
axs[0].legend()

shift = 1100
axs[1].plot(gas_age_model_regrid2, d15n_data_regrid2, 'k--',  label='$\delta^{15}$N$_2$ data', linewidth=0.5)
axs[1].fill_between(gas_age_model_regrid2, d15n_data_lo_regrid2, d15n_data_up_regrid2, alpha=0.2, facecolor='k')
axs[1].plot(gas_age_model_regrid + shift, d15n_model * 1000, 'r-', label='Goujon shifted (1100yr)', linewidth=0.9)
axs[1].plot(gas_age_model2_regrid + shift, d15n_model_up * 1000, 'r-.', label='Goujon $\Delta$T = +3K', linewidth=0.9, alpha=0.5)
axs[1].plot(gas_age_model3_regrid + shift, d15n_model_lo * 1000, 'r:', label='Goujon $\Delta$T = -3K', linewidth=0.9, alpha=0.5)
axs[1].fill_between(gas_age_model_regrid + shift, d15n_model_lo * 1000, d15n_model_up * 1000, alpha=0.2, facecolor='r')
axs[1].grid(linestyle='--', color='gray', lw='0.5')
axs[1].set_ylabel(r"$\delta^{15}$N$_2$ [‰]")
axs[1].set_xlabel(r"gas age GICC05modelext [yr]")
axs[1].legend()
plt.legend()
plt.show()

