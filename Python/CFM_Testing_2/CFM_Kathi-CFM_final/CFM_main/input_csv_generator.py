"""
This little program is being used to generate periodic input temperature or accumulation signals

inputs:
    -   start: start time of run
    -   end: end time of run
    -   delta_t: wavelength of signal
    -   ampl: amplitude of the signal
    -   x: temperature around which the sin oscillates

outputs:
    -   signal
    -   at the end csv file, saved in CFMinput

saves the signal as csv in the CFMinput folder
"""

import numpy as np


def periodicT(start, end, delta_t, ampl, T_0):
    times = np.arange(start, end + 1, 1)
    x = np.arange(0, abs(end-start) + 1, 1)
    temp_signal = ampl * np.sin(x * 1 / (2 * np.pi * delta_t)) + T_0
    signal = np.zeros([2, np.shape(temp_signal)[0]])
    signal[0] = times
    signal[1] = temp_signal
    return signal


#temp_s = periodicT(-10000, 0, 50, 15, 242.05)
#np.savetxt('CFMinput/periodic_T_50y_15K.csv', temp_s, delimiter=",")


def const_periodicT(start, end, delta_t, ampl, T_0, lag):
    times = np.arange(start, end + 1, 1)
    x = np.arange(0, abs(end-start) - lag + 1, 1)
    temp_signal = ampl * np.sin(x * (1 / delta_t) * 2 * np.pi) + T_0
    signal = np.zeros([2, np.shape(times)[0]])
    signal[0] = times
    signal[1, :lag] = T_0
    signal[1, lag:] = temp_signal
    return signal

temp_s2 = const_periodicT(-10000, 0, 500, 15, 242.05, 3000)
np.savetxt('CFMinput/periodic_T_500y_15K.csv', temp_s2, delimiter=",")



def AccTrelation(temp_signal):
    acc_signal = np.ones_like(temp_signal)
    acc_signal[0] = temp_signal[0]
    acc_signal[1] = np.exp(-21.492 + 0.0811 * temp_signal[1])
    return acc_signal

acc_s = AccTrelation(temp_s2)

np.savetxt('CFMinput/periodic_T_acc_500y_15K.csv', acc_s, delimiter=",")