from __future__ import division
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
from matplotlib import animation
import matplotlib.pyplot as plt
import h5py
import time
import os.path
import string
import shutil
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class CfmPlotter():

    def __init__(self, fpath=None):

        self.fpath = fpath
        f = h5py.File(fpath)
        # self.fs = os.path.split(fpath)[1]
        self.depth = f["depth"][:]
        self.climate = f["Modelclimate"][:]                                 # for temperature and accumulation forcing
        self.model_time = np.array(([a[0] for a in self.depth[:]]))             # model time
        self.close_off_depth = f["BCO"][:, 2]                               # close-off depth
        self.close_off_age = f["BCO"][:, 1]
        self.rho = f["density"][:]

        # these are the variables that have to be picked at COD:
        self.temperature = f["temperature"][:]
        self.temperature_cod = np.ones_like(self.close_off_depth)
        self.age = f["age"][:]
        self.delta_age = np.ones_like(self.close_off_depth)
        self.d15n = f["d15N2"][:] - 1.
        self.d15n_cod = np.ones_like(self.close_off_depth)
        self.d40ar = f["d40Ar"][:] - 1.
        self.d40ar_cod = np.ones_like(self.close_off_depth)
        for i in range(self.depth.shape[0]):
            index = int(np.where(self.depth[i, 1:] == self.close_off_depth[i])[0])
            self.temperature_cod[i] = self.temperature[i, index]
            self.d15n_cod[i] = self.d15n[i, index]
            self.d40ar_cod[i] = self.d40ar[i, index]
            self.delta_age[i] = self.age[i, index] - self.close_off_age[i]
        self.delta_temp = self.climate[:, 2] - self.temperature_cod

        f.close()
        return

    def plotting(self):
        self.f0 = plt.figure(num=0, figsize=(11, 7), dpi=200)
        self.f0.tight_layout(pad=2.8)
        self.ax01 = plt.subplot2grid((2, 3), (0, 0))
        self.ax02 = plt.subplot2grid((2, 3), (0, 1))
        self.ax03 = plt.subplot2grid((2, 3), (1, 0))
        self.ax04 = plt.subplot2grid((2, 3), (1, 1))
        self.ax05 = plt.subplot2grid((2, 3), (0, 2))
        self.ax06 = plt.subplot2grid((2, 3), (1, 2))

        # self.ax01.set_ylim((250, 0))
        # self.ax02.set_ylim((250, 0))
        # self.ax03.set_ylim((220, 245))
        # self.ax04.set_ylim((0.04, 0.2))
        # self.ax02.set_xlim((220, 245))
        # self.ax03.set_xlim((self.model_time[0], self.model_time[-1]))
        # self.ax04.set_xlim((self.model_time[0], self.model_time[-1]))
        # self.ax05.set_ylim((250, 0))
        # self.ax06.set_ylim((100, 70))
        # self.ax06.set_xlim((self.model_time[0], self.model_time[-1]))

        self.ax01.set_ylabel(r"Temperature Forcing [K]")
        self.ax01.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)  # labelpad=-1
        self.ax02.set_ylabel(r"Close-off depth [m]")
        self.ax02.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax03.set_ylabel(r"Accumulation Forcing [$\mathrm{my}^{-1}$ ice eq.]")
        self.ax03.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax04.set_ylabel(r"$\delta^{15}$N$_2$ / $\delta ^{40}$Ar/4")
        self.ax04.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax05.set_ylabel(r"Temperature Gradient [K]")
        self.ax05.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax06.set_ylabel(r"$\Delta$ age")
        self.ax06.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)

        self.ax01.set_title('Temperature Forcing', fontsize=11)
        self.ax02.set_title('Close-off Depth', fontsize=11)
        self.ax03.set_title('Accumulation Forcing', fontsize=11)
        self.ax04.set_title('Isotope Signal', fontsize=11)
        self.ax05.set_title('Temperature Gradient', fontsize=11)
        self.ax06.set_title('$\Delta$ age', fontsize=11)

        self.p011, = self.ax01.plot(self.model_time, self.climate[:, 2], 'b-')
        self.p012, = self.ax02.plot(self.model_time, self.close_off_depth, 'k-')
        self.ax02.invert_yaxis()
        self.p021, = self.ax03.plot(self.model_time, self.climate[:, 1], 'k-')
        self.p022, = self.ax04.plot(self.model_time, self.d15n_cod, 'k-', label='$\delta^{15}$N$_2$')
        self.p0221, = self.ax04.plot(self.model_time, self.d40ar_cod/4., 'b-', label='$\delta ^{40}$Ar/4')
        self.ax04.legend()
        self.p031, = self.ax05.plot(self.model_time, self.delta_temp, 'r-')
        self.p032, = self.ax06.plot(self.model_time, self.close_off_age, 'b-')

        plt.tight_layout()
        # TODO: added plt.tight_layout; this helps a bit, however labels and headers still overlap
        # self.f0.suptitle("\n".join(['Model Time: 10000y, t = 500y, Amplitude: 15K']), y=1.05)
        plt.savefig('results/ramps/double_ramp_T_50y_500y.pdf')
        plt.show()
        return


plots = CfmPlotter('results/ramps/CFMresults_T_double_ramp_50y_500y_HLdynamic.hdf5')
plots.plotting()