import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

S_PER_YEAR = 31557600.0

def find_index_from_year(time, year):
    year_ind = np.min(np.where(time >= year))
    return year_ind


def read_data_at_secondSpin(path_model, path_spin, t_second_spin):
    s = h5py.File(path_spin, 'r')
    f = h5py.File(path_model, 'r')
    time = f['depth'][1:, 0]
    t_second_spin_ind = find_index_from_year(time, t_second_spin)
    dict_SecondSpin = {
        'ageSpin': s['ageSpin'][:],
        'densitySpin': s['densitySpin'][:],
        'depthSpin': s['depthSpin'][:],
        'tempSpin': s['tempSpin'][:],
        'r2Spin': s['r2Spin'][:],
        'ageSpin2': f['age'][t_second_spin_ind, :] * S_PER_YEAR,
        'densitySpin2': f['density'][t_second_spin_ind, :],
        'depthSpin2': f['depth'][t_second_spin_ind, :],
        'tempSpin2': f['temperature'][t_second_spin_ind, :],
        'diffusivitySpin2': f['diffusivity'][t_second_spin_ind, 1:],
        'gas_ageSpin2': f['gas_age'][t_second_spin_ind, 1:],
        'w_airSpin2': f['w_air'][t_second_spin_ind, 1:],
        'w_firnSpin2': f['w_firn'][t_second_spin_ind, 1:],
        'd15N2Spin2': f['d15N2'][t_second_spin_ind, 1:],
        'r2Spin2': f['r2'][t_second_spin_ind, :]
    }
    f.close()
    s.close()
    return dict_SecondSpin


def write_data_2_new_spinFile(path_spin, dict_SecondSpin):
    with h5py.File(path_spin, 'w') as f:
        for key in dict_SecondSpin:
            f[key] = dict_SecondSpin[key][:]
    f.close()
    return


if __name__ == '__main__':
    print(glob.glob('../../CFM_main/resultsFolder/*.hdf5')[1])
    model_path = str(glob.glob('../../CFM_main/resultsFolder/*.hdf5')[1])
    # spin_path = glob.glob('resultsFolder/*.hdf5')[1]
    # model_path = '../../CFM_main/resultsFolder/CFMresults_NGRIP_Barnola_50_35kyr_300m_2yr_instant_acc.hdf5'
    spin_path = '../../CFM_main/resultsFolder/CFMspin_NGRIP_Barnola_50_35kyr_300m_2yr_instant_acc.hdf5'

    dict_spin = read_data_at_secondSpin(model_path, spin_path, -45500)
    write_data_2_new_spinFile(spin_path, dict_spin)


