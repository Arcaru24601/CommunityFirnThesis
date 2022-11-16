import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

os.chdir(r'../data/NGRIP-Kindler/data/')
myFiles_ = glob.glob('*.txt')


def read_txt(filePaths):
    dataset = pd.read_csv(filePaths, comment='#', header=0, encoding='ISO-8859-1', sep='\t')
    return dataset


def create_dict(myFiles):
    data_dict = {}
    for i in range(len(myFiles)):
        data_dict[myFiles[i].strip('.txt')] = read_txt(myFiles[i])
    return data_dict


data = create_dict(myFiles_)

# ----------------------------------------------------------------------------------------------------------------------
# temperature and accumulation - from Kindler 2014
# ----------------------------------------------------------------------------------------------------------------------

# dict_keys(['ngrip2014temp', 'ngrip2014d15n', 'ngrip2014age'])
# data['ngrip2014temp'] -  depth_m;  age-ss09sea06bm;  ageGICC05;  accum;  temp

# print(data['ngrip2014temp'])
# print(data['ngrip2014d15n'])
# print(data['ngrip2014age'])

temp = data['ngrip2014temp']['temp'] + 273.15  # convert Celsius to Kelvin
acc = data['ngrip2014temp']['accum']
age = data['ngrip2014temp']['age-ss09sea06bm ']

t = np.flipud(np.array(age)) * (-1)
temps = np.flipud(np.array(temp))
accs = np.flipud(np.array(acc))

plot = False
if plot:
    fig, ax = plt.subplots()
    ax.plot(t/1000, temps, color='blue')
    ax.set_xlabel("Age - ss09sea06bm [kyr]")
    ax.set_ylabel("Temperature [K]", color="blue")
    ax2 = ax.twinx()
    ax2.plot(t/1000, accs, color="orange")
    ax2.set_ylabel("Accumulation ice equivalent [m/yr]", color="orange")
    plt.show()

input_temps = np.array([t, temps])
input_acc = np.array([t, accs])

np.savetxt('../../../../CFM_main/CFMinput/NGRIP_T.csv', input_temps, delimiter=",")
np.savetxt('../../../../CFM_main/CFMinput/NGRIP_Acc.csv', input_acc, delimiter=",")


# ----------------------------------------------------------------------------------------------------------------------
# d15N2 - from Kindler 2014
# ----------------------------------------------------------------------------------------------------------------------

# print(data['ngrip2014d15n']) # - depth_m    age  d15N  d15Nerr
d15N = np.array(data['ngrip2014d15n']['d15N'])
d15N_err = np.array(data['ngrip2014d15n']['d15Nerr'])
depth_d15N = np.array(data['ngrip2014d15n']['depth_m'])
age = np.array(data['ngrip2014d15n']['age'])

plot = False
if plot:
    plt.plot(d15N, depth_d15N)
    #plt.fill_between(d15N + d15N_err, d15N - d15N_err, depth_d15N, alpha=0.4)
    plt.xlabel('$\delta^{15}$N$_2$ [‰]')
    plt.ylabel('Depth [m]')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# d15N2 - from Huber 2006
# ----------------------------------------------------------------------------------------------------------------------

os.chdir(r'../../NGRIP-Huber/data/')
myFiles_ = glob.glob('*.txt')
# print(myFiles_) ['ngrip2006-t.txt', 'ngrip2006d15n-ch4-t.txt', 'ngrip2006d15n.txt', 'ngrip2006-ch4.txt']
data = create_dict(myFiles_)
print(list(data['ngrip2006d15n'].columns))
print(data['ngrip2006d15n'])

#d15N2 = np.array(data['ngrip2006d15n']['d15N'])
#d15N_err2 = np.array(data['ngrip2006d15n']['d15N error'])
#depth_d15N2 = np.array(data['ngrip2006d15n']['depth'])
