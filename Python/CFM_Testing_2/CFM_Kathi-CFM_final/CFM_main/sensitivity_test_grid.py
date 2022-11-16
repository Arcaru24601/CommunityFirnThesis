from plot_delta_15N_correctage import *
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# import a colormap
cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, 6)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data_path = '~/projects/Thesis/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'

cop1 = 1/200.


# Calculate the misfit/ standard deviation

def misfit(iceage, d15n2, d15n2_projected):
    return np.sqrt(1 / (np.shape(iceage)[0]) * np.sum((d15n2_projected - d15n2) ** 2))


model_path7 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_1yr.hdf5'
model_path8 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_2yr.hdf5'
model_path9 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_5yr.hdf5'
model_path10 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_10yr.hdf5'
model_path11 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_15yr.hdf5'
model_path12 = 'results/sensitivity_test_grid/sinus_temperature_forcing/CFMresults_sin200yr_HLdynamic_300m_20yr.hdf5'

# Get d15N2 using close-off depth
# ----------------------------------------------------------------------------------------------------------------------

gas_age7, d15n7, delta_age7, ice_age7 = get_model_data_d15N(model_path7, cod=True)
gas_age8, d15n8, delta_age8, ice_age8 = get_model_data_d15N(model_path8, cod=True)
gas_age9, d15n9, delta_age9, ice_age9 = get_model_data_d15N(model_path9, cod=True)
gas_age10, d15n10, delta_age10, ice_age10 = get_model_data_d15N(model_path10, cod=True)
gas_age11, d15n11, delta_age11, ice_age11 = get_model_data_d15N(model_path11, cod=True)
gas_age12, d15n12, delta_age12, ice_age12 = get_model_data_d15N(model_path12, cod=True)

cod7, ice_age72 = get_cod(model_path7, cod=True)
cod8 = get_cod(model_path8, cod=True)[0]
cod9 = get_cod(model_path9, cod=True)[0]
cod10 = get_cod(model_path10, cod=True)[0]
cod11 = get_cod(model_path11, cod=True)[0]
cod12 = get_cod(model_path12, cod=True)[0]

IceAgeD15N = interpolate.interp1d(ice_age7, d15n7, 'linear', fill_value='extrapolate')

d15n7_8 = IceAgeD15N(ice_age8)
sigma7_8 = misfit(ice_age8, d15n8, d15n7_8)
d15n7_9 = IceAgeD15N(ice_age9)
sigma7_9 = misfit(ice_age9, d15n9, d15n7_9)
d15n7_10 = IceAgeD15N(ice_age10)
sigma7_10 = misfit(ice_age10, d15n10, d15n7_10)
d15n7_11 = IceAgeD15N(ice_age11)
sigma7_11 = misfit(ice_age11, d15n11, d15n7_11)
d15n7_12 = IceAgeD15N(ice_age12)
sigma7_12 = misfit(ice_age12, d15n12, d15n7_12)

sigma = np.array([sigma7_8, sigma7_9, sigma7_10, sigma7_11, sigma7_12])
grid_time = np.array([2, 5, 10, 15, 20])

fig, axs = plt.subplots(3, sharex=False, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(8)

axs[0].plot(ice_age7[ice_age7 > -2000], d15n7[ice_age7 > -2000] * 1000, label='1 yr', color=cmap(cmap_intervals[0]))
# axs[0].plot(ice_age72[ice_age72 > -2000], d15n7[ice_age72 > -2000] * 1000, label='1 yr', color='orange')
axs[0].plot(ice_age8[ice_age8 > -2000], d15n8[ice_age8 > -2000] * 1000, label='2 yr', color=cmap(cmap_intervals[1]))
axs[0].plot(ice_age9[ice_age9 > -2000], d15n9[ice_age9 > -2000] * 1000, label='5 yr', color=cmap(cmap_intervals[2]))
axs[0].plot(ice_age10[ice_age10 > -2000], d15n10[ice_age10 > -2000] * 1000, label='10 yr', color=cmap(cmap_intervals[3]))
axs[0].plot(ice_age11[ice_age11 > -2000], d15n11[ice_age11 > -2000] * 1000, label='15 yr', color=cmap(cmap_intervals[4]))
axs[0].plot(ice_age12[ice_age12 > -2000], d15n12[ice_age12 > -2000] * 1000, label='20 yr', color=cmap(cmap_intervals[5]))
axs[0].set_xlabel('Ice age [yr]')
axs[0].set_ylabel('$\delta^{15}$N [‰]')
axs[0].grid(linestyle='--', color='gray', lw='0.5')
axs[0].legend()

axs[1].plot(ice_age7[ice_age7 > -2000], cod7[ice_age7 > -2000], label='1 yr', color=cmap(cmap_intervals[0]))
# axs[1].plot(ice_age72[ice_age72 > -2000], cod7[ice_age72 > -2000], label='1 yr', color='orange')
axs[1].plot(ice_age8[ice_age8 > -2000], cod8[ice_age8 > -2000], label='2 yr', color=cmap(cmap_intervals[1]))
axs[1].plot(ice_age9[ice_age9 > -2000], cod9[ice_age9 > -2000], label='5 yr', color=cmap(cmap_intervals[2]))
axs[1].plot(ice_age10[ice_age10 > -2000], cod10[ice_age10 > -2000], label='10 yr', color=cmap(cmap_intervals[3]))
axs[1].plot(ice_age11[ice_age11 > -2000], cod11[ice_age11 > -2000], label='15 yr', color=cmap(cmap_intervals[4]))
axs[1].plot(ice_age12[ice_age12 > -2000], cod12[ice_age12 > -2000], label='20 yr', color=cmap(cmap_intervals[5]))
axs[1].set_xlabel('Ice age [yr]')
axs[1].set_ylabel('Close-off (Martinerie) depth [m]')
axs[1].grid(linestyle='--', color='gray', lw='0.5')
axs[1].legend()

axs[2].plot(grid_time, sigma * 1000, 'bo')
axs[2].grid(linestyle='--', color='gray', lw='0.5')
axs[2].set_xlabel('Time resolution [yr]')
axs[2].set_ylabel('$\sigma$ of $\delta^{15}$N [‰]')
plt.savefig('grid_res_sensitivity_HLdynamic_sin200yr_cod.pdf')
plt.show()


# Get d15N2 using the lock-in depth
# ----------------------------------------------------------------------------------------------------------------------
gas_age7, d15n7, delta_age7, ice_age7 = get_model_data_d15N(model_path7, cod=False)
gas_age8, d15n8, delta_age8, ice_age8 = get_model_data_d15N(model_path8, cod=False)
gas_age9, d15n9, delta_age9, ice_age9 = get_model_data_d15N(model_path9, cod=False)
gas_age10, d15n10, delta_age10, ice_age10 = get_model_data_d15N(model_path10, cod=False)
gas_age11, d15n11, delta_age11, ice_age11 = get_model_data_d15N(model_path11, cod=False)
gas_age12, d15n12, delta_age12, ice_age12 = get_model_data_d15N(model_path12, cod=False)

cod7, ice_age72 = get_cod(model_path7, cod=False)
cod8 = get_cod(model_path8, cod=False)[0]
cod9 = get_cod(model_path9, cod=False)[0]
cod10 = get_cod(model_path10, cod=False)[0]
cod11 = get_cod(model_path11, cod=False)[0]
cod12 = get_cod(model_path12, cod=False)[0]

IceAgeD15N = interpolate.interp1d(ice_age7, d15n7, 'linear', fill_value='extrapolate')

d15n7_8 = IceAgeD15N(ice_age8)
sigma7_8 = misfit(ice_age8, d15n8, d15n7_8)
d15n7_9 = IceAgeD15N(ice_age9)
sigma7_9 = misfit(ice_age9, d15n9, d15n7_9)
d15n7_10 = IceAgeD15N(ice_age10)
sigma7_10 = misfit(ice_age10, d15n10, d15n7_10)
d15n7_11 = IceAgeD15N(ice_age11)
sigma7_11 = misfit(ice_age11, d15n11, d15n7_11)
d15n7_12 = IceAgeD15N(ice_age12)
sigma7_12 = misfit(ice_age12, d15n12, d15n7_12)

sigma = np.array([sigma7_8, sigma7_9, sigma7_10, sigma7_11, sigma7_12])
grid_time = np.array([2, 5, 10, 15, 20])

fig, axs = plt.subplots(3, sharex=False, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(8)

axs[0].plot(ice_age7[ice_age7 > -2000], d15n7[ice_age7 > -2000] * 1000, label='1 yr', color=cmap(cmap_intervals[0]))
# axs[0].plot(ice_age72[ice_age72 > -2000], d15n7[ice_age72 > -2000] * 1000, label='1 yr', color='orange')
axs[0].plot(ice_age8[ice_age8 > -2000], d15n8[ice_age8 > -2000] * 1000, label='2 yr', color=cmap(cmap_intervals[1]))
axs[0].plot(ice_age9[ice_age9 > -2000], d15n9[ice_age9 > -2000] * 1000, label='5 yr', color=cmap(cmap_intervals[2]))
axs[0].plot(ice_age10[ice_age10 > -2000], d15n10[ice_age10 > -2000] * 1000, label='10 yr', color=cmap(cmap_intervals[3]))
axs[0].plot(ice_age11[ice_age11 > -2000], d15n11[ice_age11 > -2000] * 1000, label='15 yr', color=cmap(cmap_intervals[4]))
axs[0].plot(ice_age12[ice_age12 > -2000], d15n12[ice_age12 > -2000] * 1000, label='20 yr', color=cmap(cmap_intervals[5]))
axs[0].set_xlabel('Ice age [yr]')
axs[0].set_ylabel('$\delta^{15}$N [‰]')
axs[0].grid(linestyle='--', color='gray', lw='0.5')
axs[0].legend()

axs[1].plot(ice_age7[ice_age7 > -2000], cod7[ice_age7 > -2000], label='1 yr', color=cmap(cmap_intervals[0]))
# axs[1].plot(ice_age72[ice_age72 > -2000], cod7[ice_age72 > -2000], label='1 yr', color='orange')
axs[1].plot(ice_age8[ice_age8 > -2000], cod8[ice_age8 > -2000], label='2 yr', color=cmap(cmap_intervals[1]))
axs[1].plot(ice_age9[ice_age9 > -2000], cod9[ice_age9 > -2000], label='5 yr', color=cmap(cmap_intervals[2]))
axs[1].plot(ice_age10[ice_age10 > -2000], cod10[ice_age10 > -2000], label='10 yr', color=cmap(cmap_intervals[3]))
axs[1].plot(ice_age11[ice_age11 > -2000], cod11[ice_age11 > -2000], label='15 yr', color=cmap(cmap_intervals[4]))
axs[1].plot(ice_age12[ice_age12 > -2000], cod12[ice_age12 > -2000], label='20 yr', color=cmap(cmap_intervals[5]))
axs[1].set_xlabel('Ice age [yr]')
axs[1].set_ylabel('Lock-in depth (Martinerie) [m]')
axs[1].grid(linestyle='--', color='gray', lw='0.5')
axs[1].legend()

axs[2].plot(grid_time, sigma * 1000, 'bo')
axs[2].grid(linestyle='--', color='gray', lw='0.5')
axs[2].set_xlabel('Time resolution [yr]')
axs[2].set_ylabel('$\sigma$ of $\delta^{15}$N [‰]')
plt.savefig('grid_res_sensitivity_HLdynamic_sin200yr_liz.pdf')
plt.show()


# Get d15N2 using the diffusivity
# ----------------------------------------------------------------------------------------------------------------------

ice_age_diff7, d15n_diff7, d_diff7 = get_diff(model_path7)
ice_age_diff8, d15n_diff8, d_diff8 = get_diff(model_path8)
ice_age_diff9, d15n_diff9, d_diff9 = get_diff(model_path9)
ice_age_diff10, d15n_diff10, d_diff10 = get_diff(model_path10)
ice_age_diff11, d15n_diff11, d_diff11 = get_diff(model_path11)
ice_age_diff12, d15n_diff12, d_diff12 = get_diff(model_path12)

IceAgeD15N_diff = interpolate.interp1d(ice_age_diff7, d15n_diff7, 'linear', fill_value='extrapolate')

d15n_diff7_8 = IceAgeD15N_diff(ice_age_diff8)
sigma_diff7_8 = misfit(ice_age_diff8, d15n_diff8, d15n_diff7_8)
d15n_diff7_9 = IceAgeD15N_diff(ice_age_diff9)
sigma_diff7_9 = misfit(ice_age_diff9, d15n_diff9, d15n_diff7_9)
d15n_diff7_10 = IceAgeD15N_diff(ice_age_diff10)
sigma_diff7_10 = misfit(ice_age_diff10, d15n_diff10, d15n_diff7_10)
d15n_diff7_11 = IceAgeD15N_diff(ice_age_diff11)
sigma_diff7_11 = misfit(ice_age_diff11, d15n_diff11, d15n_diff7_11)
d15n_diff7_12 = IceAgeD15N_diff(ice_age_diff12)
sigma_diff7_12 = misfit(ice_age_diff12, d15n_diff12, d15n_diff7_12)

sigma_diff = np.array([sigma_diff7_8, sigma_diff7_9, sigma_diff7_10, sigma_diff7_11, sigma_diff7_12])
grid_time = np.array([2, 5, 10, 15, 20])


fig, axs = plt.subplots(3, sharex=False, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(8)

axs[0].plot(ice_age_diff7[ice_age_diff7 > -2000], d15n_diff7[ice_age_diff7 > -2000] * 1000, label='1 yr', color=cmap(cmap_intervals[0]))
axs[0].plot(ice_age_diff8[ice_age_diff8 > -2000], d15n_diff8[ice_age_diff8 > -2000] * 1000, label='2 yr', color=cmap(cmap_intervals[1]))
axs[0].plot(ice_age_diff9[ice_age_diff9 > -2000], d15n_diff9[ice_age_diff9 > -2000] * 1000, label='5 yr', color=cmap(cmap_intervals[2]))
axs[0].plot(ice_age_diff10[ice_age_diff10 > -2000], d15n_diff10[ice_age_diff10 > -2000] * 1000, label='10 yr', color=cmap(cmap_intervals[3]))
axs[0].plot(ice_age_diff11[ice_age_diff11 > -2000], d15n_diff11[ice_age_diff11 > -2000] * 1000, label='15 yr', color=cmap(cmap_intervals[4]))
axs[0].plot(ice_age_diff12[ice_age_diff12 > -2000], d15n_diff12[ice_age_diff12 > -2000] * 1000, label='20 yr', color=cmap(cmap_intervals[5]))
axs[0].set_xlabel('Ice age [yr]')
axs[0].set_ylabel('$\delta^{15}$N [‰]')
axs[0].grid(linestyle='--', color='gray', lw='0.5')
axs[0].legend()

axs[1].plot(ice_age_diff7[ice_age_diff7 > -2000], d_diff7[ice_age_diff7 > -2000], label='1 yr', color=cmap(cmap_intervals[0]))
axs[1].plot(ice_age_diff8[ice_age_diff8 > -2000], d_diff8[ice_age_diff8 > -2000], label='2 yr', color=cmap(cmap_intervals[1]))
axs[1].plot(ice_age_diff9[ice_age_diff9 > -2000], d_diff9[ice_age_diff9 > -2000], label='5 yr', color=cmap(cmap_intervals[2]))
axs[1].plot(ice_age_diff10[ice_age_diff10 > -2000], d_diff10[ice_age_diff10 > -2000], label='10 yr', color=cmap(cmap_intervals[3]))
axs[1].plot(ice_age_diff11[ice_age_diff11 > -2000], d_diff11[ice_age_diff11 > -2000], label='15 yr', color=cmap(cmap_intervals[4]))
axs[1].plot(ice_age_diff12[ice_age_diff12 > -2000], d_diff12[ice_age_diff12 > -2000], label='20 yr', color=cmap(cmap_intervals[5]))
axs[1].set_xlabel('Ice age [yr]')
axs[1].set_ylabel('Depth (D eff = 0) [m]')
axs[1].grid(linestyle='--', color='gray', lw='0.5')
axs[1].legend()

axs[2].plot(grid_time, sigma_diff * 1000, 'bo')
axs[2].grid(linestyle='--', color='gray', lw='0.5')
axs[2].set_xlabel('Time resolution [yr]')
axs[2].set_ylabel('$\sigma$ of $\delta^{15}$N [‰]')
plt.savefig('grid_res_sensitivity_HLdynamic_sin200yr_diff.pdf')
plt.show()


# Calculate misfit between data and fit

model_path1 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_1yr.hdf5'
model_path2 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_2yr.hdf5'
model_path3 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_5yr.hdf5'
model_path4 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_10yr.hdf5'
model_path5 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_15yr.hdf5'
model_path6 = 'results/sensitivity_test_grid/CFMresults_NGRIP_HLdynamic_65_30kyr_300m_20yr.hdf5'
 
a1 = get_model_data_d15N_from_diff(model_path1, cod=True)
a2 = get_model_data_d15N_from_diff(model_path2, cod=True)
a3 = get_model_data_d15N_from_diff(model_path3, cod=True)
a4 = get_model_data_d15N_from_diff(model_path4, cod=True)
a5 = get_model_data_d15N_from_diff(model_path5, cod=True)
a6 = get_model_data_d15N_from_diff(model_path6, cod=True)

b1 = get_icecore_d15N_data(data_path, cop1, a1[0], a1[2], a1[3])
b2 = get_icecore_d15N_data(data_path, cop1, a2[0], a2[2], a2[3])
b3 = get_icecore_d15N_data(data_path, cop1, a3[0], a3[2], a3[3])
b4 = get_icecore_d15N_data(data_path, cop1, a4[0], a4[2], a4[3])
b5 = get_icecore_d15N_data(data_path, cop1, a5[0], a5[2], a5[3])
b6 = get_icecore_d15N_data(data_path, cop1, a6[0], a6[2], a6[3])

plot_data = True
if plot_data:
    plt.plot(a1[3], a1[1] * 1000, label='1 yr', color=cmap(cmap_intervals[0]))
    plt.plot(a2[3], a2[1] * 1000, label='2 yr', color=cmap(cmap_intervals[1]))
    plt.plot(a3[3], a3[1] * 1000, label='5 yr', color=cmap(cmap_intervals[2]))
    plt.plot(a4[3], a4[1] * 1000, label='10 yr', color=cmap(cmap_intervals[3]))
    plt.plot(a5[3], a5[1] * 1000, label='15 yr', color=cmap(cmap_intervals[4]))
    plt.plot(a6[3], a6[1] * 1000, label='20 yr', color=cmap(cmap_intervals[5]))
    plt.plot(a1[3], b1, 'k', label='data')
    plt.plot()
    #plt.plot(a6[3], b2, label='data2')
    plt.xlabel('GICC05modelext ice age [yr]')
    plt.ylabel('$\delta^{15}$N [‰]')
    plt.legend()
    plt.savefig('grid_res_test_NGRIP_45-30kyr_HLdynamic.pdf')
    plt.show()


'''
sigma1 = misfit(a1[3], a1[1], b1)
sigma2 = misfit(a2[3], a2[1], b2)
sigma3 = misfit(a3[3], a3[1], b3)
sigma4 = misfit(a4[3], a4[1], b4)
sigma5 = misfit(a5[3], a5[1], b5)
sigma6 = misfit(a6[3], a6[1], b6)

print(sigma1, sigma2, sigma3, sigma4, sigma5, sigma6)
'''



