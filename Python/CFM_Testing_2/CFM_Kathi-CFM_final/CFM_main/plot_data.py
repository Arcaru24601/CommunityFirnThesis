import matplotlib.pyplot as plt
import numpy as np
from read_hdf5 import getDataFromKey

'''
OUTPUTS:

- depth: (m) The depth of each model node. 
- density: (kg m-3) The density at the depths in ‘depth 
- temperature: (K) Temperature at the depths in ‘depth’ 
- age: (years) Firn Age at the depths in ‘depth’ 
- dcon: Dcon is a layer-tracking routine; to use it you need to dig into the code a bit and program it how you want, 
    but for example you could set it up so that each model node that has liquid water gets a 1 and all others get 
    a zero. Corresponds to depth.
- bdot_mean: (m a-1 ice equivalent) the mean accumulation rate over the lifetime of each parcel of firn, corresponds 
    with ‘depth’
- climate: The temperature (K) and accumulation rate (m a-1 ice equivalent) at each time step – useful if using 
    interpolation to find determine the climate.
- compaction: (m) Total compaction of each node since the previous time step; corresponds to ‘depth’. To get 
    compaction rate you need to divide by the time-step size. To get compaction over an interval you need to sum numerous 
    boxes.
- grainsize: (mm2) the grain size of the firn, corresponds to ‘depth’ temp_Hx: the temperature history of the firn 
    (See Morris and Wingham, 2014)
- isotopes: (per mil) water isotope values, corresponds to ‘depth’
- LWC: (m3) volume of liquid present in that node, corresponds to ‘depth’
- DIP: the depth-integrated porosity and change in surface elevation. 4 columns: The first is time, second is DIP to 
    the bottom of the model domain (m), third is change in domain thickness since last time step (m), fourth is change 
    in domain thickness since start of model run (m). DIP also saves a variable called DIPc, which is a matrix of the 
    cumulative porosity to the depth in ‘depth’
- BCO: bubble close-off properties. 10 columns: time, Martinerie close-off age, Marinerie close-off depth, age of 
    830 kg m-3 density horizon, depth of 830 kg m-3 density horizon, Martinerie lock-in age, Marinerie lock-in depth, 
    age of 815 kg m-3 density horizon, depth of 815 kg m-3 density horizon, depth of zero porosity.
- FirnAir: only works if FirnAir is true in the config.json. Saves gas concentrations, diffusivity profile, gas age, 
    and advection rates of air and firn, all corresponding to ‘depth’.

'''

def loadDataFromResults(filenames):
    ages = []
    depths = []
    densities = []
    temperatures = []
    for i in range(len(filenames)):
        ages.append(np.array(getDataFromKey(filenames[i], 'age')))
        depths.append(np.array(getDataFromKey(filenames[i], 'depth')))
        densities.append(np.array(getDataFromKey(filenames[i], 'density')))
        temperatures.append(np.array(getDataFromKey(filenames[i], 'temperature')))
    return ages, depths, densities, temperatures

def find_idx_nearest(array, t):     # needs the first column of the first entry in temps list as input
    idx = (np.abs(array - t)).argmin()
    return idx

# should work for temperature, age & density (... and other parameters, still need to try that...)
def getData_at_t(data, t):
    data_t = []
    for i in range(len(data)):
        idx = find_idx_nearest(data[i][:, 0], t)
        data_t.append(data[i][idx, 1:])
    return data_t

def plotTemperatureAndDensity2Depth(depth, density, temperature, age, savename, model):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Density and temperature profiles (2020)')
    for i in range(len(model)):
        ax1.plot(depth[i][1:], density[i], 'o', markersize=0.5, label=model[i])
        ax2.plot(depth[i][1:], temperature[i], 'o', markersize=0.5, label=model[i])
        ax3.plot(depth[i][1:], age[i], 'o', markersize=0.5, label=model[i])
    ax1.set(xlabel='depth [m]')
    ax2.set(xlabel='depth [m]')
    ax3.set(xlabel='depth [m]')
    ax1.set(ylabel='density [kg/m³]')
    ax2.set(ylabel='temperature [K]')
    ax3.set(ylabel='age [yr]')
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.tight_layout()
    savepath = '../../figures/'
    plt.savefig(savepath + savename, format='pdf')
    return plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# PLOT RESULTS ---------------------------------------------------------------------------------------------------------

# steady state NGRIP
'''
filename1 = "../../results/steady_state_NGRIP/CFMresults_steady_state_Barnola1991.hdf5"
filename2 = "../../results/steady_state_NGRIP/CFMresults_steady_state_Goujon2003.hdf5"
filename3 = "../../results/steady_state_NGRIP/CFMresults_steady_state_HLSigfus.hdf5"
filename4 = "../../results/steady_state_NGRIP/CFMresults_steady_state_HLdynamic.hdf5"

filenames = [filename1, filename2, filename3, filename4]
model = ['Barnola1991', 'Goujon2003', 'HLSigfus', 'HLdynamics']

ages, depths, densities, temps = loadDataFromResults(filenames)

age_t = getData_at_t(ages, 2000)
density_t = getData_at_t(densities, 2000)
temps_t = getData_at_t(temps, 2000)

plotTemperatureAndDensity2Depth(depths, density_t, temps_t, age_t, 'rho_T_vs_depth_NGRIP2020', model)
'''

# ramp NGRIP
'''
filename1 = "../../results/ramp_NGRIP/CFMresults_ramp_Barnola1991.hdf5"
filename2 = "../../results/ramp_NGRIP/CFMresults_ramp_Goujon2003.hdf5"
filename3 = "../../results/ramp_NGRIP/CFMresults_ramp_HLSigfus.hdf5"
filename4 = "../../results/ramp_NGRIP/CFMresults_ramp_HLdynamic.hdf5"

filenames = [filename1, filename2, filename3, filename4]
model = ['Barnola1991', 'Goujon2003', 'HLSigfus', 'HLdynamics']

ages, depths, densities, temps = loadDataFromResults(filenames)

age_t = getData_at_t(ages, 2000)
density_t = getData_at_t(densities, 2000)
temps_t = getData_at_t(temps, 2000)

plotTemperatureAndDensity2Depth(depths, density_t, temps_t, age_t, 'rho_T_vs_depth_ramp_NGRIP2020', model)
'''


# steady state Dome C
filename1 = "../../results/steady_state_DomeC/CFMresults_steady_state_Barnola1991.hdf5"
filename2 = "../../results/steady_state_DomeC/CF
age_t = getData_at_t(ages, 2000)
density_t = getData_at_t(densities, 2000)
temps_t = getData_at_t(temps, 2000)

#plotTemperatureAndDensity2Depth(depths, density_t, temps_t, age_t, 'rho_T_vs_depth_steady_state_DomeC2020', model)

Mresults_steady_state_Goujon2003.hdf5"
filename3 = "../../results/steady_state_DomeC/CFMresults_steady_state_HLSigfus.hdf5"
filename4 = "../../results/steady_state_DomeC/CFMresults_steady_state_HLdynamic.hdf5"

filenames = [filename1, filename2, filename3, filename4]
model = ['Barnola1991', 'Goujon2003', 'HLSigfus', 'HLdynamics']

ages, depths, densities, temps = loadDataFromResults(filenames)
print("boo")
print(temps[0])

age_t = getData_at_t(ages, 2000)
density_t = getData_at_t(densities, 2000)
temps_t = getData_at_t(temps, 2000)

#plotTemperatureAndDensity2Depth(depths, density_t, temps_t, age_t, 'rho_T_vs_depth_steady_state_DomeC2020', model)


# ramp Dome C
'''
filename1 = "../../results/ramp_DomeC/CFMresults_ramp_Barnola1991.hdf5"
filename2 = "../../results/ramp_DomeC/CFMresults_ramp_Goujon2003.hdf5"
filename3 = "../../results/ramp_DomeC/CFMresults_ramp_HLSigfus.hdf5"
filename4 = "../../results/ramp_DomeC/CFMresults_ramp_HLdynamic.hdf5"

filenames = [filename1, filename2, filename3, filename4]
model = ['Barnola1991', 'Goujon2003', 'HLSigfus', 'HLdynamics']

ages, depths, densities, temps = loadDataFromResults(filenames)

age_t = getData_at_t(ages, 2000)
density_t = getData_at_t(densities, 2000)
temps_t = getData_at_t(temps, 2000)

plotTemperatureAndDensity2Depth(depths, density_t, temps_t, age_t, 'rho_T_vs_depth_ramp_DomeC2020', model)
'''
