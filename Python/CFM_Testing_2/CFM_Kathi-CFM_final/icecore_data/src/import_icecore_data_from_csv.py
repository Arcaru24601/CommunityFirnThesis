import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csaps

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ----------------------------------------------------------------------------------------------------------------------
# Import temperature and accumulation from NGRIP Excel sheets

file_location = '~/projects/Thesis/CFM_Kathi/icecore_data/data/NGRIP/supplement.xlsx'
df1 = pd.read_excel(file_location, sheet_name='Sheet4')
df2 = pd.read_excel(file_location, sheet_name='Sheet5')
df3 = pd.read_excel(file_location, sheet_name='Sheet6')

start_year = 50000
end_year = 30000

age = np.array(df1[df1.columns[2]])

t_start_ind = np.min(np.where(age >= start_year))
t_end_ind = np.min(np.where(age >= end_year))

t = np.flipud(age[t_end_ind:t_start_ind]) * (-1)
accs = np.flipud(np.array(df1[df1.columns[3]])[t_end_ind:t_start_ind])
temps = np.flipud(np.array(df1[df1.columns[4]])[t_end_ind:t_start_ind])

temps_err = np.flipud(np.array(df1[df1.columns[5]])[t_end_ind:t_start_ind])
temps_lo = temps - temps_err
temps_up = temps + temps_err

# ----------------------------------------------------------------------------------------------------------------------
# Make cubic smoothing spline

cop1 = 1/300.  # cut-off period


def smooth_parameter(cop, age_):
    dx = np.mean(np.diff(age_))  # mean distance between two points
    lamda = (1 / (2 * cop * np.pi)) ** 4 / dx  # eg. 8 in Enting1987
    p = 1 / (1 + lamda)  # close to eq. 13 in Enting1987
    return p


p1 = smooth_parameter(cop1, age)

sp1 = csaps.CubicSmoothingSpline(t, temps, smooth=p1)
sp2 = csaps.CubicSmoothingSpline(t, accs, smooth=p1)

t_grid = np.arange(-start_year, -end_year + 20, 20)
temp_smooth = sp1(t_grid)
accs_smooth = sp2(t_grid)

temp_smooth_lo = temp_smooth - 3.  # -3K


# ----------------------------------------------------------------------------------------------------------------------
# Save in CFM format
input_temps = np.array([t_grid, temp_smooth_lo])
# input_temps_lo = np.array([t, temps_lo])
# input_temps_up = np.array([t, temps_up])
input_acc = np.array([t_grid, accs_smooth])

np.savetxt('../../CFM_main/CFMinput/NGRIP_T_65_30kyr.csv', input_temps, delimiter=",")
# np.savetxt('../../CFM_main/CFMinput/NGRIP_T_lo_100kyr.csv', input_temps_lo, delimiter=",")
# np.savetxt('../../CFM_main/CFMinput/NGRIP_T_up_100kyr.csv', input_temps_up, delimiter=",")
np.savetxt('../../CFM_main/CFMinput/NGRIP_Acc_65_30kyr.csv', input_acc, delimiter=",")


# ----------------------------------------------------------------------------------------------------------------------
# Plot temperature and accumulation
plot = True
if plot:
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
fig, axs = plt.subplots(2)
axs[0].plot(t, temps)

axs[1].plot(age, temps0)
plt.show()


fig, axs = plt.subplots(2)
axs[0].plot(age[t_end_ind:t_start_ind]/ 1000, np.array(df1[df1.columns[4]])[t_end_ind:t_start_ind])
axs[1].plot(t / 1000, temps)
plt.show()
'''