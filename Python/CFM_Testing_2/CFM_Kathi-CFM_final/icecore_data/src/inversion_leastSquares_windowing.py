from scipy.optimize import least_squares
from read_d18O import *
from read_d15N import *
from read_temp_acc import *
from calculate_d15N import *
from secondSpin import read_data_at_secondSpin, write_data_2_new_spinFile, find_index_from_year
import os
import json
import glob
import math

# ----------------------------------------------------------------------------------------------------------------------
# Data & Model paths

data_path = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/interpolated_data.xlsx'
data_path2 = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'

resultsFileName_Spin = 'CFMresults_NGRIP_Barnola_5000kyr_window_550m_2yr_inversion-LS_SPIN2_2022-08-01_01.hdf5'
resultsFileName_Main = 'CFMresults_NGRIP_Barnola_110-10kyr_550m_2yr_inversion-LS_MAIN_2022-08-01_01.hdf5'

spin2_path = '../../CFM_main/resultsFolder/' + resultsFileName_Spin
model_path = '../../CFM_main/resultsFolder/' + resultsFileName_Main

finalResults_path_modelruns = '~/projects/finalResults/inversion/Goujon_windowing_LS_2022-08-12_01/'

json_SPIN = 'FirnAir_NGRIP_Barnola_windowing.json'
json_MAIN = 'FirnAir_NGRIP_Spin2_Barnola_windowing.json'

# optimization parameter files
results_minimizer_spin_path = 'resultsFolder/2022-06-16_01_resultsInversion_minimizer_SPIN.h5'
results_minimizer_main_path = 'resultsFolder/2022-06-16_01_resultsInversion_minimizer.h5'

# ----------------------------------------------------------------------------------------------------------------------
# Set parameters
window_width = 5000
step_width = 500
start_year_0 = -120000  # start input year for the actual run (main run)
end_year_0 = start_year_0 + window_width  # end input year for the actual run (main run)
end_year_M = -10000
M = int((abs(end_year_M - start_year_0) - window_width) / step_width + 1)  # number of windows
year_Spin = 3000  # Years of first Spin (with constant temperature and accumulation)
year_Spin2 = 9000  # Years of second Spin
overlap = 2000
start_year_Spin2 = start_year_0 - (year_Spin2 - overlap)
end_year_Spin2 = start_year_0 + overlap
time_linear_temp = 200  # first years of main run, where I interpolate from spin temperature to main run temperature
# forcing
print('M ', M)

firnair_module = True  # this is to specify whether we use the firnair module in the CFM

stpsPerYear = 0.5
S_PER_YEAR = 31557600.0

physRho_option = 'Goujon2003'  # 'HLdynamic', 'Goujon2003', 'HLSigfus', 'Barnola1991'

cop_ = 1 / 200.  # cut-off frequency for cubic smoothing spline (low pass filter)
time_grid_stp_ = 20  # step length time grid --> also for cubic smoothing spline
cod_mode = 'lid'  # 'cod', 'lid', '0_diff'

optimizer = 'minimize'  # 'least_squares', 'minimize'
method = 'Nelder-Mead'  # 'BFGS', 'Nelder-Mead'
theta_0 = [0.37, 73]  # initial guess
N = 1000  # number of max iterations

d15n_age = 'ice_age'  # 'gas_age', 'ice_age'  NOTE: Until now 'gas_age' only works if firnair_module=True !!!
frac_minimizer_interval = 0.5  # fraction of ice_age/d15N interval where optimization is performed
no_points_minimize_Spin = 5  # first points of ice_age/d15N interval where spin optimization is performed

# ----------------------------------------------------------------------------------------------------------------------
# Read d18O data from NGRIP
# -----------------------------

depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
temp, temp_err = read_temp(data_path)
acc = read_acc(data_path)

# For Spin run ---------------------------------------------------------------------------------------------------------
depth_interval_Spin, d18O_interval_Spin, ice_age_interval_Spin = get_interval_data_noTimeGrid(depth_full, d18O_full,
                                                                                              ice_age_full,
                                                                                              start_year_Spin2,
                                                                                              end_year_Spin2)
d18O_interval_perm_Spin = d18O_interval_Spin * 1000
d18o_smooth_Spin = smooth_data(1 / 200., d18O_interval_perm_Spin, ice_age_interval_Spin, ice_age_interval_Spin)[0]

t = 1. / theta_0[0] * d18o_smooth_Spin + theta_0[1]

temp, temp_err = read_temp(data_path)
temp_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year_Spin2, end_year_Spin2)[0]
# plt.plot(ice_age_interval_Spin, t)
# plt.plot(ice_age_interval_Spin, temp_interval)
# plt.show()


temp_interval_Spin = get_interval_temp(temp, temp_err, ice_age_full, start_year_Spin2, end_year_Spin2)[0]
acc_interval_Spin = get_interval_acc(acc, ice_age_full, start_year_Spin2, end_year_Spin2)
input_acc_Spin = np.array([ice_age_interval_Spin, acc_interval_Spin])
np.savetxt('../../CFM_main/CFMinput/optimize_acc.csv', input_acc_Spin, delimiter=",")

years_Spin = (np.max(ice_age_interval_Spin) - np.min(ice_age_interval_Spin)) * 1.0
dt_Spin = S_PER_YEAR / stpsPerYear  # seconds per time step
stp = int(years_Spin * S_PER_YEAR / dt_Spin)  # -1       # total number of time steps, as integer
modeltime_Spin = np.linspace(start_year_Spin2, end_year_Spin2, stp + 1)[:-1]

opt_dict_Spin = {'count_Spin': np.zeros([N, 1], dtype=int),
                 'a_Spin': np.zeros([N, 1]),
                 'b_Spin': np.zeros([N, 1]),
                 'c_Spin': np.zeros([N, 1]),
                 'd_Spin': np.zeros([N, 1]),
                 'd15N@cod_Spin': np.zeros([N, np.shape(modeltime_Spin)[0]]),
                 'd15N_data_Spin': np.zeros([N, np.shape(modeltime_Spin)[0]]),
                 'd15N_data_err_Spin': np.zeros([N, np.shape(modeltime_Spin)[0]]),
                 'ice_age_Spin': np.zeros([N, np.shape(modeltime_Spin)[0]]),
                 'gas_age_Spin': np.zeros([N, np.shape(modeltime_Spin)[0]]),
                 'cost_function_Spin': np.zeros([N, 1])}

# For Main run ---------------------------------------------------------------------------------------------------------

opt_dict = {'count': np.zeros([N, 1], dtype=int),
            'a': np.zeros([N, 1]),
            'b': np.zeros([N, 1]),
            'c': np.zeros([N, 1]),
            'd': np.zeros([N, 1]),
            'd15N@cod': np.zeros([N, window_width]),
            'd15N_data': np.zeros([N, window_width]),
            'd15N_data_err': np.zeros([N, window_width]),
            'ice_age': np.zeros([N, window_width]),
            'gas_age': np.zeros([N, window_width]),
            'cost_function': np.zeros([N, 1])}

# ----------------------------------------------------------------------------------------------------------------------
# Define the function to optimize
# ---------------------------------
def fun_Spin(theta):
    count = int(np.max(opt_dict_Spin['count_Spin']))
    print('iteration', count)

    a = theta[0]
    b = theta[1]
    print('a: %s, b: %s' % (a, b))

    temperature_Spin = 1. / a * d18o_smooth_Spin + b
    input_temperature_Spin = np.array([ice_age_interval_Spin, temperature_Spin])
    print(ice_age_interval_Spin[0], ice_age_interval_Spin[-1])
    np.savetxt('../../CFM_main/CFMinput/optimize_T.csv', input_temperature_Spin, delimiter=",")
    os.chdir('../../CFM_main/')
    os.system('python3 main.py %s -n' % json_SPIN)

    if os.path.exists('resultsFolder/%s' % resultsFileName_Spin):
        os.chdir('../icecore_data/src/')
        d15N2_model_, iceAge_model_, gasAge_model_, deltaAge_ = get_d15N_model(spin2_path, mode=cod_mode,
                                                                               firnair=firnair_module, cop=1 / 200.)

        ice_age_data_interv, gas_age_data_interv, d15N2_data_interv, d15N2_data_err_interv = \
            get_d15N_data_interval(data_path2, iceAge_model_)

        d15N2_model_interp, gasAge_model_interp = interpolate_d15Nmodel_2_d15Ndata(d15N2_model_, iceAge_model_,
                                                                                   gasAge_model_, ice_age_data_interv)
        shape_optimize = 33 # int(np.shape(d15N2_model_interp)[0] / 2)

        cost_func = 1 / (np.shape(d15N2_model_interp[-shape_optimize:])[0] - 1) \
                    * np.sum(((d15N2_model_interp[-shape_optimize:] - d15N2_data_interv[-shape_optimize:])
                              / d15N2_data_err_interv[-shape_optimize:]) ** 2)
        if math.isnan(cost_func):
            linear = np.linspace(0, -30, 33)
            print(linear)
            residuals = np.ones(33) * 0.8 * linear
        else:
            residuals = (d15N2_model_interp[-shape_optimize:] - d15N2_data_interv[-shape_optimize:])#\
                    #/ d15N2_data_err_interv[-shape_optimize:]
        print('d15nmodel: ', d15N2_model_interp[-shape_optimize:])
        print('d15n_data: ', d15N2_data_interv[-shape_optimize:])
        print('d15n_err: ', d15N2_data_err_interv[-shape_optimize:])

        opt_dict_Spin['d15N@cod_Spin'][count, :np.shape(d15N2_model_interp)[0]] = d15N2_model_interp[:]
        opt_dict_Spin['d15N_data_Spin'][count, :np.shape(d15N2_data_interv)[0]] = d15N2_data_interv[:]
        opt_dict_Spin['d15N_data_err_Spin'][count, :np.shape(d15N2_data_err_interv)[0]] = d15N2_data_err_interv[:]
        opt_dict_Spin['ice_age_Spin'][count, :np.shape(d15N2_model_interp)[0]] = ice_age_data_interv[:]
        opt_dict_Spin['gas_age_Spin'][count, :np.shape(d15N2_model_interp)[0]] = gasAge_model_interp[:]

    else:
        os.chdir('../icecore_data/src/')
        cost_func = 800.
        print('------------------------------------------------------------------------------------------')
        print('<<<<<<<< Close-off crashed everything again - Setting cost function to 800! >>>>>>>>>>>>>>')
        print('------------------------------------------------------------------------------------------')
        linear = np.linspace(0, 100, 33)
        print(linear)
        residuals = np.ones(33) * 0.8 * linear

    opt_dict_Spin['a_Spin'][count] = a
    opt_dict_Spin['b_Spin'][count] = b

    opt_dict_Spin['cost_function_Spin'][count] = cost_func
    print('cost function Spin: ', cost_func)
    count += 1
    opt_dict_Spin['count_Spin'][count] = count
    print(residuals)
    return residuals


def fun(theta):
    count = int(np.max(opt_dict['count']))
    print('iteration', count)

    a = theta[0]
    b = theta[1]
    print('a: %s, b: %s' % (a, b))
    temperature = 1. / a * d18o_smooth + b
    # plt.plot(ice_age_interval, temperature)
    # plt.show()
    if count != 0:
        print('count is not 0 - calculating linear temperature interval')
        index_N = find_index_from_year(ice_age_interval, ice_age_interval[0] + time_linear_temp)
        temp_N = temperature[index_N]
        temp_0 = 1. / opt_dict['a'][0] * d18o_smooth[0] + opt_dict['b'][0]
        time_N = ice_age_interval[index_N]
        time_0 = ice_age_interval[0]
        delta_T = temp_N - temp_0
        delta_t = time_N - time_0
        for i in range(index_N + 1):
            temperature[i] = temp_0 + delta_T / delta_t * (ice_age_interval[i] - time_0)

    input_temperature = np.array([ice_age_interval, temperature])

    np.savetxt('../../CFM_main/CFMinput/optimize_T.csv', input_temperature, delimiter=",")
    os.chdir('../../CFM_main/')
    os.system('python3 main.py %s' % json_MAIN)

    if os.path.exists('resultsFolder/%s' % resultsFileName_Main):
        print('Created output file!')
        os.chdir('../icecore_data/src/')
        d15N2_model_, iceAge_model_, gasAge_model_, deltaAge_ = get_d15N_model(model_path, mode=cod_mode,
                                                                               firnair=firnair_module, cop=1 / 200.)

        ice_age_data_interv, gas_age_data_interv, d15N2_data_interv, d15N2_data_err_interv = \
            get_d15N_data_interval(data_path2, iceAge_model_)
        print('shape data: ', np.shape(gas_age_data_interv))

        d15N2_model_interp, gasAge_model_interp = interpolate_d15Nmodel_2_d15Ndata(d15N2_model_, iceAge_model_,
                                                                                   gasAge_model_, ice_age_data_interv)
        #index_minimize = find_index_from_year(ice_age_data_interv, ice_age_data_interv[0] + 2 * time_linear_temp)
        index_minimize = 25
        print('index minimize: ', index_minimize)
        if np.shape(gas_age_data_interv)[0] < index_minimize:
            linear = np.linspace(0, 100, 25)
            residuals = np.ones(25) * 0.8 * linear
            cost_func = 800.
        else:
            cost_func = 1 / (np.shape(d15N2_model_interp[-index_minimize:])[0] - 1) \
                        * np.sum(((d15N2_model_interp[-index_minimize:] - d15N2_data_interv[-index_minimize:])
                                  / d15N2_data_err_interv[-index_minimize:]) ** 2)
            if math.isnan(cost_func):
                linear = np.linspace(0, 100, 25)
                residuals = np.ones(25) * 0.8 * linear
            else:
                residuals = (d15N2_model_interp[-25:] - d15N2_data_interv[-25:])\
                    / d15N2_data_err_interv[-25:]

        print('shape minimize interval:', np.shape(d15N2_model_interp))
        opt_dict['d15N@cod'][count, :np.shape(d15N2_model_interp)[0]] = d15N2_model_interp[:]
        opt_dict['d15N_data'][count, :np.shape(d15N2_data_interv)[0]] = d15N2_data_interv[:]
        opt_dict['d15N_data_err'][count, :np.shape(d15N2_data_err_interv)[0]] = d15N2_data_err_interv[:]
        opt_dict['ice_age'][count, :np.shape(d15N2_model_interp)[0]] = ice_age_data_interv[:]
        opt_dict['gas_age'][count, :np.shape(d15N2_model_interp)[0]] = gasAge_model_interp[:]
        print('d15nmodel: ', d15N2_model_interp[index_minimize:])
        print('d15n_data: ', d15N2_data_interv[index_minimize:])
        print('d15n_err: ', d15N2_data_err_interv[index_minimize:])
        print(np.shape(d15N2_model_interp[-index_minimize:]), np.shape(d15N2_data_interv[-index_minimize:]),
              np.shape(d15N2_data_err_interv[-index_minimize:]))
    else:
        print('There is no output file -_- ')
        os.chdir('../icecore_data/src/')
        cost_func = 800.
        print('------------------------------------------------------------------------------------------')
        print('<<<<<<<< Close-off crashed everything again - Setting cost function to 800! >>>>>>>>>>>>>>')
        print('------------------------------------------------------------------------------------------')
        linear = np.linspace(0, 100, 25)
        residuals = np.ones(25) * 0.5 * linear

    opt_dict['a'][count] = a
    opt_dict['b'][count] = b

    opt_dict['cost_function'][count] = cost_func
    count += 1
    opt_dict['count'][count] = count
    print('cost function: ', cost_func)
    return residuals

# ----------------------------------------------------------------------------------------------------------------------
# MINIMIZE
# ----------------------

# Spin prepare json
os.chdir('../../CFM_main/')
with open(json_SPIN, 'r+') as json_file:
    cfm_params = json.load(json_file)
    # cfm_params['TWriteStart'] = start_year_Spin2
    cfm_params['yearSpin'] = year_Spin
    json_file.seek(0)
    json.dump(cfm_params, json_file, indent=4)
    json_file.truncate()
    cfm_params['physRho'] = physRho_option
    json_file.seek(0)
    json.dump(cfm_params, json_file, indent=4)
    json_file.truncate()
    # cfm_params['SecondSpin'] = False
    cfm_params['resultsFileName'] = resultsFileName_Spin
    json_file.seek(0)
    json.dump(cfm_params, json_file, indent=4)
    json_file.truncate()
    print(cfm_params['spinFileName'])
    json_file.seek(0)
    json.dump(cfm_params, json_file, indent=4)
    json_file.truncate()
    json_file.close()

# Spin run optimization

os.chdir('../icecore_data/src/')
res_Spin = least_squares(fun_Spin, theta_0, method='trf', gtol=1e-8, x_scale=[0.5, 100], diff_step=[0.01, 0.05], verbose=2)
entry_0 = np.where(opt_dict_Spin['count_Spin'] == 0)[0]
opt_dict_Spin['count_Spin'] = np.delete(opt_dict_Spin['count_Spin'], entry_0[1:])
opt_dict_Spin['count_Spin'] = opt_dict_Spin['count_Spin'][:-1]
max_int = np.shape(opt_dict_Spin['count_Spin'])[0]
with h5py.File(results_minimizer_spin_path, 'w') as f_Spin:
    for key in opt_dict_Spin:
        f_Spin[key] = opt_dict_Spin[key][:max_int]
f_Spin.close()
theta_Spin = res_Spin.x

print('----------------------------------------------')
print('|            INFO MINIMIZE SPIN              |')
print('----------------------------------------------')
print(res_Spin.message)
print(res_Spin.success)
print('Theta Spin: ', theta_Spin)

os.chdir('../../CFM_main/resultsFolder/')
model_path_2 = glob.glob('CFMresults*.hdf5')[0]
spin_path_2 = glob.glob('CFMspin*.hdf5')[0]

print('Reading Data at start year for main run ...')
dict_spin = read_data_at_secondSpin(model_path_2, spin_path_2, start_year_0)
write_data_2_new_spinFile(spin_path_2, dict_spin)

os.system('mv %s %s' % (model_path_2, finalResults_path_modelruns + model_path_2))
os.system('cp %s %s' % (spin_path_2, finalResults_path_modelruns + spin_path_2))



# ----------------------------------------------------------------------------------------------------------------------
# Main run prepare json
os.chdir('../../icecore_data/src/')

theta_Window = [0.0, 0.0]
print('theta_Window')
opt_dict_Window = {'window': np.zeros([N, 1], dtype=int),
                   'a': np.zeros([N, 1]),
                   'b': np.zeros([N, 1]),
                   'c': np.zeros([N, 1]),
                   'd': np.zeros([N, 1]),
                   'd15N@cod': np.zeros([N, window_width]),
                   'd15N_data': np.zeros([N, window_width]),
                   'd15N_data_err': np.zeros([N, window_width]),
                   'ice_age': np.zeros([N, window_width]),
                   'gas_age': np.zeros([N, window_width]),
                   'cost_function': np.zeros([N, 1])}

for i in range(M):
    print(i)
    start_year_ = start_year_0 + i * step_width
    end_year_ = start_year_ + window_width
    print('Window ', i)
    print('start year: %5.3f, end year: %5.3f' % (start_year_, end_year_))
    depth_interval, d18O_interval, ice_age_interval = get_interval_data_noTimeGrid(depth_full, d18O_full,
                                                                                   ice_age_full,
                                                                                   start_year_, end_year_)
    d18O_interval_perm = d18O_interval * 1000
    d18o_smooth = smooth_data(1 / 200., d18O_interval_perm, ice_age_interval, ice_age_interval)[0]

    temp_interval = get_interval_temp(temp, temp_err, ice_age_full, start_year_, end_year_)[0]
    acc_interval = get_interval_acc(acc, ice_age_full, start_year_, end_year_)
    input_acc = np.array([ice_age_interval, acc_interval])
    np.savetxt('../../CFM_main/CFMinput/optimize_acc_main.csv', input_acc, delimiter=",")
    os.chdir('../../CFM_main/')
    with open(json_MAIN, 'r+') as json_file:
        cfm_params = json.load(json_file)
        cfm_params['TWriteStart'] = start_year_
        json_file.seek(0)
        json.dump(cfm_params, json_file, indent=4)
        json_file.truncate()
        cfm_params['physRho'] = physRho_option
        json_file.seek(0)
        json.dump(cfm_params, json_file, indent=4)
        json_file.truncate()
        # cfm_params['SecondSpin'] = True
        cfm_params['resultsFileName'] = resultsFileName_Main
        json_file.seek(0)
        json.dump(cfm_params, json_file, indent=4)
        json_file.truncate()

    # Main run optimization
    os.chdir('../icecore_data/src/')
    if i == 0:
        res_Main = least_squares(fun, theta_Spin, method='trf', gtol=1e-8, x_scale=[0.5, 100], diff_step=[0.01, 0.05], verbose=2)
    else:
        res_Main = least_squares(fun, theta_Spin, method='trf', gtol=1e-8, x_scale=[0.5, 100], diff_step=[0.01, 0.05],
                                 verbose=2)
    entry_0 = np.where(opt_dict['count'] == 0)[0]
    opt_dict['count'] = np.delete(opt_dict['count'], entry_0[1:])
    opt_dict['count'] = opt_dict['count'][:-1]
    max_int = np.shape(opt_dict['count'])[0] - 1

    theta_Window = res_Main.x

    print('----------------------------------------------')
    print('|            INFO MINIMIZE MAIN              |')
    print('----------------------------------------------')
    print(res_Main.message)
    print(res_Main.success)
    print('Theta Main: ', theta_Window)

    os.chdir('../../CFM_main/')
    model_path_2 = glob.glob('resultsFolder/CFMresults*.hdf5')[0]
    spin_path_2 = glob.glob('resultsFolder/CFMspin*.hdf5')[0]

    print('Reading Data at start year for next window main run ...')
    dict_spin = read_data_at_secondSpin(model_path_2, spin_path_2, start_year_)
    write_data_2_new_spinFile(spin_path_2, dict_spin)

    print('a', opt_dict['a'][max_int])
    print(opt_dict['a'])

    opt_dict_Window['window'][i] = i
    opt_dict_Window['a'][i] = opt_dict['a'][max_int]
    opt_dict_Window['b'][i] = opt_dict['b'][max_int]
    opt_dict_Window['c'][i] = opt_dict['c'][max_int]
    opt_dict_Window['d'][i] = opt_dict['d'][max_int]
    opt_dict_Window['d15N@cod'][i, :np.shape(opt_dict['d15N@cod'][-1, :])[0]] = opt_dict['d15N@cod'][max_int, :]
    opt_dict_Window['d15N_data'][i, :np.shape(opt_dict['d15N@cod'][-1, :])[0]] = opt_dict['d15N_data'][max_int, :]
    opt_dict_Window['d15N_data_err'][i, :np.shape(opt_dict['d15N@cod'][-1, :])[0]] = opt_dict['d15N_data_err'][max_int, :]
    opt_dict_Window['ice_age'][i, :np.shape(opt_dict['d15N@cod'][-1, :])[0]] = opt_dict['ice_age'][max_int, :]
    opt_dict_Window['gas_age'][i, :np.shape(opt_dict['d15N@cod'][-1, :])[0]] = opt_dict['gas_age'][max_int, :]
    opt_dict_Window['cost_function'][i] = opt_dict['cost_function'][max_int]

    opt_dict = {'count': np.zeros([N, 1], dtype=int),
                'a': np.zeros([N, 1]),
                'b': np.zeros([N, 1]),
                'c': np.zeros([N, 1]),
                'd': np.zeros([N, 1]),
                'd15N@cod': np.zeros([N, window_width]),
                'd15N_data': np.zeros([N, window_width]),
                'd15N_data_err': np.zeros([N, window_width]),
                'ice_age': np.zeros([N, window_width]),
                'gas_age': np.zeros([N, window_width]),
                'cost_function': np.zeros([N, 1])}

    os.chdir('../icecore_data/src/')
with h5py.File(results_minimizer_main_path, 'w') as f:
    for key in opt_dict_Window:
        f[key] = opt_dict_Window[key][:]  # only saving the last step
    f.close()
os.chdir('resultsFolder/')
os.system('mv *.h5 %s' % finalResults_path_modelruns)
