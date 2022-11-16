import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_T_surface(file):
    T_surface = file['temperature'][:, 1]
    return T_surface


def get_T_cod(file, mode):
    depth = file['depth'][:]
    cod = get_cod(file, mode)
    T = file['temperature'][:]
    T_cod = np.ones_like(cod)
    for i in range(depth.shape[0]):
        index = int(np.where(depth[i, 1:] == cod[i])[0])
        T_cod[i] = T[i, index]
    return T_cod


def T_mean(file, mode):
    T_surface = get_T_surface(file)
    T_cod = get_T_cod(file, mode)
    T_means = np.zeros_like(T_cod)
    for i in range(np.shape(T_surface)[0]):
        if T_surface[i] > T_cod[i]:
            T_means[i] = (T_cod[i] * T_surface[i])/(T_surface[i] - T_cod[i]) * np.log(T_surface[i] / T_cod[i])
        else:
            T_means[i] = (T_cod[i] * T_surface[i])/(T_cod[i] - T_surface[i]) * np.log(T_cod[i] / T_surface[i])
    return T_means


def get_cod(file, mode):
    if mode == 'cod':         # Get data at Martinerie close-off depth
        cod = file["BCO"][:, 2]

    if mode == 'lid':         # Get data at LID
        cod = file["BCO"][:, 6]

    if mode == '0_diff':       # Get data at depth where D_eff = 0
        diffusivity = file['diffusivity'][:]
        depth_model = file['depth']
        cod = np.zeros(np.shape(diffusivity)[0])
        index = np.zeros(np.shape(diffusivity)[0])
        for i in range(np.shape(diffusivity)[0]):
            index[i] = np.max(np.where(diffusivity[i, 1:] > 10 ** (-20))) + 1
            cod[i] = depth_model[i, int(index[i])]
    return cod


def d15N_therm(file, mode):
    T_means = T_mean(file, mode)
    T_surface = get_T_surface(file)
    T_cod = get_T_cod(file, mode)
    alpha = (8.656 - 1232 / T_means) * 1 / 1000
    d15N_thermal = ((T_surface / T_cod) ** alpha - 1) * 1000
    return d15N_thermal


def d15N_grav(file, mode):
    T_means = T_mean(file, mode)
    cod = get_cod(file, mode)
    GRAVITY = 9.81
    R = 8.3145
    delta_M = 1 / 1000
    d15N_gravitation = (np.exp((delta_M * GRAVITY * cod) / (R * T_means)) - 1) * 1000
    return d15N_gravitation


def d15N_tot(file, mode):
    d15N_thermal = d15N_therm(file, mode)
    d15N_gravitation = d15N_grav(file, mode)
    return d15N_thermal + d15N_gravitation


if __name__ == '__main__':
    model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_50_35kyr_300m_2yr_instant_acc.hdf5'
    mode = 'lid'
    f = h5py.File(model_path, 'r')
    T_S = get_T_surface(f)
    T_c = get_T_cod(f, mode)
    plt.plot(T_S, label='T_S')
    plt.plot(T_c, label='T_cod')
    plt.plot(T_mean(f, mode), label='T_mean')
    plt.legend()
    plt.show()

    d15n_therm = d15N_therm(f, mode)
    d15n_grav = d15N_grav(f, mode)
    d15n_tot = d15N_tot(f, mode)
    plt.plot(d15n_therm, label='thermal')
    plt.plot(d15n_grav, label='gravitation')
    plt.plot(d15n_tot, label='total')
    plt.legend()
    plt.show()
