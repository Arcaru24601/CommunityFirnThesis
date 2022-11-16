import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from smoothing_splines import *
import h5py
import matplotlib.pyplot as plt
from read_d18O import get_interval_data_noTimeGrid, find_start_end_index
from calculate_d15N import *


def get_d15N_data(path_data):
    # read d15N data
    df = pd.read_excel(path_data, sheet_name='Sheet6')
    depth_d15n = np.flipud(np.array(df[df.columns[2]]))
    d15n = np.flipud(np.array(df[df.columns[3]]))
    d15n_err = np.flipud(np.array(df[df.columns[4]]))

    # ice_age and corresponding depth
    df2 = pd.read_excel(path_data, sheet_name='Sheet5')
    ice_age_scales = np.flipud(np.array(df2[df2.columns[3]])) * (-1)
    gas_age_scales = np.flipud(np.array(df2[df2.columns[4]])) * (-1)
    depth_scales = np.flipud(np.array(df2[df2.columns[0]]))

    # Interpolate to d15n
    IAD = interpolate.interp1d(depth_scales, ice_age_scales, 'linear', fill_value='extrapolate')
    ice_age_d15n = IAD(depth_d15n)

    GAD = interpolate.interp1d(depth_scales, gas_age_scales, 'linear', fill_value='extrapolate')
    gas_age_d15n = GAD(depth_d15n)

    return ice_age_d15n, gas_age_d15n, d15n, d15n_err


def get_d15N_data_interval(path_data, ice_age_d15N_model):
    # Data from ice core
    # ---------------------------------------------------------------------------------------------------
    ice_age_d15n, gas_age_d15n, d15n, d15n_err = get_d15N_data(path_data)
    start_year = np.min(ice_age_d15N_model)
    end_year = np.max(ice_age_d15N_model)

    start_ind, end_ind = find_start_end_index(ice_age_d15n, start_year, end_year)
    ice_age_d15n_interval, gas_age_d15n_interval, d15n_interval, d15n_err_interval = \
        ice_age_d15n[start_ind:end_ind], gas_age_d15n[start_ind:end_ind], d15n[start_ind:end_ind], \
        d15n_err[start_ind:end_ind]

    return ice_age_d15n_interval, gas_age_d15n_interval, d15n_interval, d15n_err_interval


def interpolate_d15Nmodel_2_d15Ndata(d15n_model, ice_age_model, gas_age_model, ice_age_data):
    Data_Model = interpolate.interp1d(ice_age_model, d15n_model, 'linear', fill_value='extrapolate')
    d15n_model_interp_ice_age = Data_Model(ice_age_data)
    IA_GA = interpolate.interp1d(ice_age_model, gas_age_model, 'linear', fill_value='extrapolate')
    gas_age_model_interp = IA_GA(ice_age_data)
    return d15n_model_interp_ice_age, gas_age_model_interp


def get_d15N_model(path_model, mode, firnair, cop):
    # Data from model
    # ---------------------------------------------------------------------------------------------------
    file = h5py.File(path_model, 'r')
    depth_model = file['depth'][:]
    ice_age_model = file["age"][:]

    if firnair:
        if mode == 'cod':
            # Get data at Martinerie close-off depth
            close_off_depth = file["BCO"][:, 2]
            gas_age_model = file["gas_age"][:]
            d15n_model = file["d15N2"][:] - 1.
            d15n_cod = np.ones_like(close_off_depth)
            gas_age_cod = np.ones_like(close_off_depth)
            ice_age_cod = np.ones_like(close_off_depth)

        if mode == 'lid':
            # Get data at LID
            close_off_depth = file["BCO"][:, 6]
            gas_age_model = file["gas_age"][:]
            d15n_model = file["d15N2"][:] - 1.
            d15n_cod = np.ones_like(close_off_depth)
            gas_age_cod = np.ones_like(close_off_depth)
            ice_age_cod = np.ones_like(close_off_depth)

        if mode == '0_diff':
            # Get data at depth where D_eff = 0
            diffusivity = file['diffusivity'][:]
            gas_age_model = file["gas_age"][:]
            d15n_model = file["d15N2"][:] - 1.
            index = np.zeros(np.shape(diffusivity)[0])
            ice_age_cod = np.zeros(np.shape(diffusivity)[0])
            gas_age_cod = np.zeros(np.shape(diffusivity)[0])
            d15n_cod = np.zeros(np.shape(diffusivity)[0])
            close_off_depth = np.zeros(np.shape(diffusivity)[0])
            for i in range(np.shape(diffusivity)[0]):
                index[i] = np.max(np.where(diffusivity[i, 1:] > 10 ** (-20))) + 1
                ice_age_cod[i] = ice_age_model[i, int(index[i])]
                d15n_cod[i] = d15n_model[i, int(index[i])]
                close_off_depth[i] = depth_model[i, int(index[i])]

        for i in range(depth_model.shape[0]):
            index = int(np.where(depth_model[i, 1:] == close_off_depth[i])[0])
            gas_age_cod[i] = gas_age_model[i, index]
            ice_age_cod[i] = ice_age_model[i, index]
            d15n_cod[i] = d15n_model[i, index]

        gas_age_cod = gas_age_cod[1:]
        ice_age_cod = ice_age_cod[1:]
        modeltime = depth_model[1:, 0]

        gas_age_cod_smooth = smooth_data(cop, gas_age_cod, modeltime, modeltime)[0]
        ice_age_cod_smooth = smooth_data(cop, ice_age_cod, modeltime, modeltime)[0]

        ice_age = modeltime - ice_age_cod_smooth  # signs seem to be the wrong way round because of negative modeltime
        delta_age = ice_age_cod_smooth - gas_age_cod_smooth
        gas_age = ice_age + delta_age
        d15n_cod = d15n_cod[1:] * 1000

    else:
        d15n_cod = d15N_tot(file=file, mode=mode)
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


'''
def get_d15N_data(path_data, ice_age_d15N_model):
    # Data from ice core
    # ---------------------------------------------------------------------------------------------------
    df = pd.read_excel(path_data)
    depth_data = np.flipud(np.array(df[df.columns[0]]))
    d15n_data = np.flipud(np.array(df[df.columns[9]]))
    d15n_err = np.flipud(np.array(df[df.columns[10]]))
    ice_age_data = np.flipud(np.array(df[df.columns[1]])) * (-1)

    # Interpolate ice_age from d15N2 data to the ice_age from d15N_model
    # ---------------------------------------------------------------------------------------------------
    Data2Model = interpolate.interp1d(ice_age_data, depth_data, 'linear', fill_value='extrapolate')
    depth_regrid = Data2Model(ice_age_d15N_model)

    # get corresponding d15N values
    # ---------------------------------------------------------------------------------------------------
    ND = interpolate.interp1d(depth_data, d15n_data, 'linear', fill_value='extrapolate')
    d15n_data_regrid = ND(depth_regrid)
    ND_err = interpolate.interp1d(depth_data, d15n_err, 'linear', fill_value='extrapolate')
    d15n_err_regrid = ND_err(depth_regrid)

    # Apply cubic smoothing spline to d15N data
    # ---------------------------------------------------------------------------------------------------
    # ice_age_d15N_model = remove_negative_values(ice_age_d15N_model)  # remove values with negative np.diff(ice_age)
    # d15n_smooth = smooth_data(cop, d15n_data_regrid, ice_age_d15N_model, ice_age_d15N_model)[0]
    # d15n_err_smooth = smooth_data(cop, d15n_err_regrid, ice_age_d15N_model, ice_age_d15N_model)[0]

    return d15n_data_regrid, d15n_err_regrid


def get_d15N_data_gasage(path_data, ice_age_d15N_model):
    # Data from ice core
    # ---------------------------------------------------------------------------------------------------
    df = pd.read_excel(path_data)
    depth_data = np.flipud(np.array(df[df.columns[0]]))
    d15n_data = np.flipud(np.array(df[df.columns[9]]))
    d15n_err = np.flipud(np.array(df[df.columns[10]]))
    gas_age_data = np.flipud(np.array(df[df.columns[2]])) * (-1)

    # Interpolate ice_age from d15N2 data to the gas_age from d15N_model
    # ---------------------------------------------------------------------------------------------------
    Data2Model = interpolate.interp1d(gas_age_data, depth_data, 'linear', fill_value='extrapolate')
    depth_regrid = Data2Model(ice_age_d15N_model)

    # get corresponding d15N values
    # ---------------------------------------------------------------------------------------------------
    ND = interpolate.interp1d(depth_data, d15n_data, 'linear', fill_value='extrapolate')
    d15n_data_regrid = ND(depth_regrid)
    ND_err = interpolate.interp1d(depth_data, d15n_err, 'linear', fill_value='extrapolate')
    d15n_err_regrid = ND_err(depth_regrid)

    # Apply cubic smoothing spline to d15N data
    # ---------------------------------------------------------------------------------------------------
    # ice_age_d15N_model = remove_negative_values(gas_age_d15N_model)  # remove values with negative np.diff(ice_age)
    # d15n_smooth = smooth_data(cop, d15n_data_regrid, gas_age_d15N_model, gas_age_d15N_model)[0]
    # d15n_err_smooth = smooth_data(cop, d15n_err_regrid, gas_age_d15N_model, gas_age_d15N_model)[0]

    return d15n_data_regrid, d15n_err_regrid
'''




# ----------------------------------------------------------------------------------------------------------------------
# Test the function
# -------------------

if __name__ == '__main__':
    # model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_50_35kyr_300m_2yr_instant_acc.hdf5'
    model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_110-10kyr_300m_2yr_inversion-NM_MAIN_2022-06-10_01.hdf5'

    data_path = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/interpolated_data.xlsx'
    data_path2 = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'

    # ice_age, gas_age, d15N, d15N_err = get_d15N_data(data_path2)
    # plt.plot(gas_age, d15N)
    # plt.show()

    d15n_model, ice_age_model, gas_age_model, delta_age = get_d15N_model(model_path, 'cod', firnair=True, cop=1/200.)
    d15n_model2, ice_age_model2, gas_age_model2, delta_age2 = get_d15N_model(model_path, 'lid', firnair=True, cop=1/200.)

    ice_age_int, gas_age_int, d15N_int, d15N_err_int = get_d15N_data_interval(data_path2, ice_age_model)
    d15N_model_interp, gasage_interp = interpolate_d15Nmodel_2_d15Ndata(d15n_model, ice_age_model, gas_age_model, ice_age_int)
    d15N_model_interp2, gasage_interp2 = interpolate_d15Nmodel_2_d15Ndata(d15n_model2, ice_age_model2, gas_age_model2, ice_age_int)


    # plt.plot(ice_age_model, d15n_model, label='model')
    plt.plot(ice_age_int, d15N_int, 'r', markersize=1, label='data')
    # plt.plot(ice_age_int, d15N_model_interp, 'b', markersize=1, label='model interpolated cod')
    # plt.plot(ice_age_int, d15N_model_interp2, 'orange', markersize=1, label='model interpolated lid')
    plt.plot(ice_age_model, d15n_model, 'b', label='cod', linewidth=1)
    plt.plot(ice_age_model2, d15n_model2, 'orange', label='lid', linewidth=1)
    plt.legend()
    plt.show()

    print(d15n_model-d15n_model2)
    # plt.plot(gas_age_int, d15N_int, 'ro', markersize=1, label='data')
    # plt.plot(gasage_interp, d15N_model_interp, 'bv', markersize=1, label='model interpolated')
    # plt.legend()
    # plt.show()


    test = False

    if test:
        # model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_65_30kyr_300m_5yr.hdf5'
        model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_110-10kyr_300m_2yr_inversion-NM_MAIN_2022-06-10_01.hdf5'
        data_path = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/interpolated_data.xlsx'

        d15N2_model, iceAge_model, gasAge_model, deltaAge = get_d15N_model(model_path, mode='cod', firnair=False, cop=1/200.)
        ice_age_data, gas_age_data, d15N2_data, d15N2_err = get_d15N_data(data_path2)

        plt.plot(iceAge_model, d15N2_model, label='model')
        plt.plot(iceAge_model, d15N2_data, 'ko', markersize=1, label='data')
        plt.plot(iceAge_model, d15N2_data + d15N2_err, 'k-.', linewidth=0.9, alpha=0.5)
        plt.plot(iceAge_model, d15N2_data - d15N2_err, 'k-.', linewidth=0.9, alpha=0.5)
        plt.fill_between(iceAge_model, d15N2_data - d15N2_err, d15N2_data + d15N2_err, alpha=0.2, facecolor='k')
        plt.xlabel('GICC05modelext ice age [yr]')
        plt.ylabel('$\delta^{15}$N$_2$ [‰]')
        plt.legend()
        plt.grid(':')
        plt.plot()
        plt.show()




