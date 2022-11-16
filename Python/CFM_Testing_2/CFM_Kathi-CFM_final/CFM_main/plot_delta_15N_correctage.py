import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import csaps


def smooth_parameter(cop, age_):
    dx = np.mean(np.diff(age_))  # mean distance between two points
    lamda = (1 / (2 * cop * np.pi)) ** 4 / dx  # eg. 8 in Enting1987
    p = 1 / (1 + lamda)  # close to eq. 13 in Enting1987
    return p


def get_model_data_d15N(path_model, cod):
    # Data from model
    # ------------------------------------------------------------------------------------------------------
    file = h5py.File(path_model, 'r')

    if cod:
        # Get data at close-off density
        close_off_depth = file["BCO"][:, 2]
        depth_model = file['depth']
        d15n = file["d15N2"][:] - 1.
        d15n_cod = np.ones_like(close_off_depth)
        gas_age_model = file["gas_age"][:]
        ice_age_model = file["age"][:]
        gas_age_cod = np.ones_like(close_off_depth)
        ice_age_cod = np.ones_like(close_off_depth)


    else:
        # Get data at LID
        close_off_depth = file["BCO"][:, 6]
        depth_model = file['depth']
        d15n = file["d15N2"][:] - 1.
        d15n_cod = np.ones_like(close_off_depth)
        gas_age_model = file["gas_age"][:]
        ice_age_model = file["age"][:]
        gas_age_cod = np.ones_like(close_off_depth)
        ice_age_cod = np.ones_like(close_off_depth)

    for i in range(depth_model.shape[0]):
        index = int(np.where(depth_model[i, 1:] == close_off_depth[i])[0])
        gas_age_cod[i] = gas_age_model[i, index]
        ice_age_cod[i] = ice_age_model[i, index]
        d15n_cod[i] = d15n[i, index]

    gas_age_cod = gas_age_cod[1:]
    ice_age_cod = ice_age_cod[1:]
    modeltime = depth_model[1:, 0]

    cop1 = 1. / 200
    p = smooth_parameter(cop1, modeltime)
    sp = csaps.CubicSmoothingSpline(modeltime, gas_age_cod, smooth=p)
    gas_age_cod_smooth = sp(modeltime)

    sp2 = csaps.CubicSmoothingSpline(modeltime, ice_age_cod, smooth=p)
    ice_age_cod_smooth = sp2(modeltime)

    ice_age = modeltime - ice_age_cod_smooth  # signs seem to be wrong way round because of negative modeltime
    delta_age = ice_age_cod_smooth - gas_age_cod_smooth
    gas_age = ice_age + delta_age

    d15n_cod = d15n_cod[1:]

    return gas_age, d15n_cod, delta_age, ice_age


def get_icecore_d15N_data(path_data, cop, gas_age, delta_age, ice_age):
    # Data from ice core
    # ---------------------------------------------------------------------------------------------------
    df3 = pd.read_excel(path_data, sheet_name='Sheet6')
    depth_data = np.flipud(np.array(df3[df3.columns[2]]))
    d15n_data = np.flipud(np.array(df3[df3.columns[3]]))

    df2 = pd.read_excel(path_data, sheet_name='Sheet5')
    gas_age_data = np.flipud(np.array(df2[df2.columns[4]])) * (-1)
    ice_age_data = np.flipud(np.array(df2[df2.columns[3]])) * (-1)
    depth = np.flipud(np.array(df2[df2.columns[0]]))

    # make a new ice age grid with mean distance 10 yr
    mean_distance = 10
    min_ice_age = np.min(ice_age)
    max_ice_age = np.max(ice_age)
    ice_age_grid = np.arange(min_ice_age, max_ice_age, mean_distance)

    # get depth from data corresponding to the modelled ice_age --> depth_regrid
    # ---------------------------------------------------------------------------------------------------
    MD = interpolate.interp1d(ice_age_data, depth, 'linear', fill_value='extrapolate')
    depth_regrid = MD(ice_age_grid)

    # get corresponding d15N values
    # ---------------------------------------------------------------------------------------------------
    ND = interpolate.interp1d(depth_data, d15n_data, 'linear', fill_value='extrapolate')
    d15n_data_regrid = ND(depth_regrid)

    # Apply cubic smoothing spline to d15N data
    # ---------------------------------------------------------------------------------------------------
    p = smooth_parameter(cop, ice_age_data)
    sp = csaps.CubicSmoothingSpline(ice_age_grid, d15n_data_regrid, smooth=p)
    d15n_smooth = sp(ice_age)

    return d15n_smooth


def get_cod(path_model, cod):
    file = h5py.File(path_model, 'r')
    if cod:
        close_off_depth = file["BCO"][:, 2]
    else:
        close_off_depth = file['BCO'][:, 6]
    depth_model = file['depth']
    ice_age_model = file["age"][:]
    ice_age_cod = np.ones_like(close_off_depth)

    for i in range(depth_model.shape[0]):
        index = int(np.where(depth_model[i, 1:] == close_off_depth[i])[0])
        ice_age_cod[i] = ice_age_model[i, index]

    ice_age_cod = ice_age_cod[1:]
    modeltime = depth_model[1:, 0]

    cop1 = 1. / 200
    p = smooth_parameter(cop1, modeltime)
    sp2 = csaps.CubicSmoothingSpline(modeltime, ice_age_cod, smooth=p)
    ice_age_cod_smooth = sp2(modeltime)
    ice_age = modeltime - ice_age_cod_smooth
    close_off_depth = close_off_depth[1:]

    return close_off_depth, ice_age


def get_diff(path_model):
    file = h5py.File(path_model, 'r')
    diffusivity = file['diffusivity'][:]
    depth_model = file['depth']
    modeltime = depth_model[1:, 0]
    ice_age_model = file["age"][:]
    d15n = file['d15N2'][:] - 1.
    index = np.zeros(np.shape(diffusivity)[0])
    ice_age_cod = np.zeros(np.shape(diffusivity)[0])
    d15n_cod = np.zeros(np.shape(diffusivity)[0])
    cod = np.zeros(np.shape(diffusivity)[0])
    for i in range(np.shape(diffusivity)[0]):
        index[i] = np.max(np.where(diffusivity[i, 1:] > 10 ** (-20))) + 1
        ice_age_cod[i] = ice_age_model[i, int(index[i])]
        d15n_cod[i] = d15n[i, int(index[i])]
        cod[i] = depth_model[i, int(index[i])]
    d15n_cod = d15n_cod[1:]

    cop1 = 1. / 200

    '''
    p = smooth_parameter(cop1, modeltime)
    sp2 = csaps.CubicSmoothingSpline(modeltime, ice_age_cod[1:], smooth=p)
    ice_age_cod_smooth = sp2(modeltime)
    ice_age = modeltime - ice_age_cod_smooth
    '''
    ice_age = modeltime - ice_age_cod[1:]
    cod = cod[1:]
    return ice_age, d15n_cod, cod


def get_model_data_d15N_from_diff(path_model, cod):
    # Data from model
    # ------------------------------------------------------------------------------------------------------
    file = h5py.File(path_model, 'r')
    diffusivity = file['diffusivity'][:]
    depth_model = file['depth']
    ice_age_model = file["age"][:]
    gas_age_model = file['gas_age'][:]
    d15n = file['d15N2'][:] - 1.
    index = np.zeros(np.shape(diffusivity)[0])
    ice_age_cod = np.zeros(np.shape(diffusivity)[0])
    gas_age_cod = np.zeros(np.shape(diffusivity)[0])
    d15n_cod = np.zeros(np.shape(diffusivity)[0])
    cod = np.zeros(np.shape(diffusivity)[0])
    for i in range(np.shape(diffusivity)[0]):
        index[i] = np.max(np.where(diffusivity[i, 1:] > 10 ** (-20))) + 1
        ice_age_cod[i] = ice_age_model[i, int(index[i])]
        gas_age_cod[i] = gas_age_model[i, int(index[i])]
        d15n_cod[i] = d15n[i, int(index[i])]
        cod[i] = depth_model[i, int(index[i])]
    d15n_cod = d15n_cod[1:]

    gas_age_cod = gas_age_cod[1:]
    ice_age_cod = ice_age_cod[1:]
    plt.plot(ice_age_cod[1:])
    plt.show()
    modeltime = depth_model[1:, 0]

    cop1 = 1. / 200
    p = smooth_parameter(cop1, modeltime)
    sp = csaps.CubicSmoothingSpline(modeltime, gas_age_cod, smooth=p)
    gas_age_cod_smooth = sp(modeltime)

    sp2 = csaps.CubicSmoothingSpline(modeltime, ice_age_cod, smooth=p)
    ice_age_cod_smooth = sp2(modeltime)

    ice_age = modeltime - ice_age_cod_smooth  # signs seem to be wrong way round because of negative modeltime
    delta_age = ice_age_cod_smooth - gas_age_cod_smooth
    gas_age = ice_age + delta_age

    return gas_age, d15n_cod, delta_age, ice_age






'''
fig, ax = plt.subplots()
ax.plot(t / 1000, temps, 'o', markersize=1, color='blue')
ax.plot(t_grid / 1000, temp_smooth, color='blue')
ax.set_xlabel("Age GICC05modelext [kyr]")
ax.set_ylabel("Temperature [K]", color="blue")
ax2 = ax.twinx()
ax2.plot(t / 1000, accs, 'o', markersize=1, color="orange")
ax2.plot(t_grid / 1000, accs_smooth, color="orange")
ax2.set_ylabel("Accumulation ice equivalent [m/yr]", color="orange")
plt.show()
'''








