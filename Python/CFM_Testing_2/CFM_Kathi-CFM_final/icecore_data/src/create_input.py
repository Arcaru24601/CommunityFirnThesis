from scipy.optimize import minimize
from read_d18O import *
from read_d15N import *
from read_temp_acc import *
import os
import json
import glob

data_path = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/interpolated_data.xlsx'
data_path2 = '~/projects/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'

start_year_ = -114500  # start input year for the actual run (main run)
end_year_ = -10000  # end input year for the actual run (main run)

depth_full, d18O_full, ice_age_full = read_data_d18O(data_path)
depth_interval, d18O_interval, ice_age_interval = get_interval_data_noTimeGrid(depth_full, d18O_full,
                                                                               ice_age_full,
                                                                               start_year_, end_year_)

acc = read_acc(data_path)
acc_interval = get_interval_acc(acc, ice_age_full, start_year_, end_year_)
input_acc = np.array([ice_age_interval, acc_interval])
np.savetxt('../../CFM_main/CFMinput/optimize_acc_try.csv', input_acc, delimiter=",")


d18O_interval_perm = d18O_interval * 1000
d18o_smooth = smooth_data(1 / 200., d18O_interval_perm, ice_age_interval, ice_age_interval)[0]

a = 0.3624298115989979
b = 66.61934161150177
temperature = 1. / a * d18o_smooth + b
input_temperature = np.array([ice_age_interval, temperature])
np.savetxt('../../CFM_main/CFMinput/optimize_T_try.csv', input_temperature, delimiter=",")

