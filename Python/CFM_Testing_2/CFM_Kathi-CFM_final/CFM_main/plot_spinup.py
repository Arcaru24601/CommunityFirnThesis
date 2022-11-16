import matplotlib.pyplot as plt
import numpy as np
from read_hdf5 import getDataFromKey
from read_hdf5 import getKeysHdf5
import h5py

filename1 = "../../results/CFMresults_ramp_Barnola1991.hdf5"
filename2 = "../../results/CFMresults_ramp_Goujon2003.hdf5"
filename3 = "../../results/CFMresults_ramp_HLSigfus.hdf5"
filename4 = "../../results/CFMresults_ramp_HLdynamic.hdf5"

filenames = [filename1, filename2, filename3, filename4]
model = ['Barnola1991', 'Goujon2003', 'HLSigfus', 'HLdynamics']

#f = h5py.File(filename1, "r")
#print("Keys: %s" % f.keys())

keys_spin_up = getKeysHdf5(filename1)
print(keys_spin_up)


def loadDataFromResults(filenames):
    ages = []
    depths = []
    densities = []
    temperatures = []
    for i in range(len(filenames)):
        ages.append(np.array(getDataFromKey(filenames[i], 'ageSpin')))
        depths.append(np.array(getDataFromKey(filenames[i], 'depthSpin')))
        densities.append(np.array(getDataFromKey(filenames[i], 'densitySpin')))
        temperatures.append(np.array(getDataFromKey(filenames[i], 'tempSpin')))
    return ages, depths, densities, temperatures

def getData_at_t(data):
    ages = data[0]
    depths = data[1]
    densities = data[2]
    temperatures = data[3]
    age_t = []
    density_t = []
    temperature_t = []
    for i in range(len(data[0])):
        agex = ages[i]
        age_t.append(agex[:])
        densityx = densities[i]
        density_t.append(densityx[:])
        temperaturex = temperatures[i]
        temperature_t.append(temperaturex[:])
    return age_t, depths, density_t, temperature_t


data = loadDataFromResults(filenames)

#age1 = np.array(age, dtype=object)

#print(np.shape(age1))
data_t = getData_at_t(data)

age_t = data_t[0]
print(len(age_t))
depths = data[1]
density_t = data[2]
temperature_t = data[3]


def plotTemperatureAndDensity2Age(age, density, temperature, savename, model):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
    #fig.suptitle('Density and temperature profiles (2020)')
    for i in range(len(model)):
        print(np.shape(age[i]))
        print(np.shape(density[i]))
        print(np.shape(temperature[i]))

        ax1.plot(age[i][-1, 1:], density[i][-1, 1:], 'o', markersize=0.5, label=model[i])
        ax2.plot(age[i][-1, 1:], temperature[i][-1, 1:], 'o', markersize=0.5, label=model[i])
    ax1.set(xlabel='age [yr]')
    ax2.set(xlabel='age [yr]')
    ax1.set(ylabel='density [kg/m³]')
    ax2.set(ylabel='temperature [K]')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    savepath = '../../figures/'
    plt.savefig(savepath + savename, format='pdf')
    return plt.show()



def plotTemperatureAndDensity2Depth(depth, density, temperature, savename, model):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
    #fig.suptitle('Density and temperature profiles (2020)')
    for i in range(len(model)):
        ax1.plot(depth[i][-1, 1:], density[i][-1, 1:], 'o', markersize=0.5, label=model[i])
        ax2.plot(depth[i][-1, 1:], temperature[i][-1, 1:], 'o', markersize=0.5, label=model[i])
    ax1.set(xlabel='depth [m]')
    ax2.set(xlabel='depth [m]')
    ax1.set(ylabel='density [kg/m³]')
    ax2.set(ylabel='temperature [K]')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    savepath = '../../figures/'
    plt.savefig(savepath + savename, format='pdf')
    return plt.show()



plotTemperatureAndDensity2Age(age_t, density_t, temperature_t, 'rho_T_vs_age_DomeC2020', model)
#plotTemperatureAndDensity2Depth(depths, density_t, temperature_t, 'rho_T_vs_depth_NGRIP2020', model)