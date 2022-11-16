from csaps import CubicSmoothingSpline
import numpy as np


def smooth_parameter(cop, x):   # cop: cut-off period
    dx = np.mean(np.diff(x))  # mean distance between points
    lamda = (1 / (2 * cop * np.pi)) ** 4 / dx  # eg. 8 in Enting1987
    p = 1 / (1 + lamda)  # close to eq. 13 in Enting1987
    return p


def smooth_data(cop, ydata2smooth, xdata, new_grid):
    p = smooth_parameter(cop, xdata)
    sp = CubicSmoothingSpline(xdata, ydata2smooth, smooth=p)
    y_smooth = sp(new_grid)
    return y_smooth, new_grid


def remove_negative_values(data):
    if np.any(np.diff(data) < 0):
        ind = np.where(np.diff(data) < 0)
        data[ind] = data[ind + 1]
        print('Changed value at %5.3f to %5.3f', (ind, ind + 1))
    return data
