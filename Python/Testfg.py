# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:41:42 2022

@author: jespe
"""

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
    #Jesper
    def __init__(self, fpath=None):
        self.fpath = fpath
        f = h5py.File(fpath)
        # self.fs = os.path.split(fpath)[1]
        self.depth = f["depth"][:]
        self.climate = f["Modelclimate"][:]  # for temperature and accumulation forcing
        self.model_time = np.array(([a[0] for a in self.depth[:]]))  # model time
        self.close_off_depth = f["BCO"][:, 2]  # close-off depth
        self.close_off_age = f["BCO"][:, 1]
        self.rho = f["density"][:]

        # these are the variables that have to be picked at COD:
        self.temperature = f["temperature"][:]
        self.temperature_cod = np.ones_like(self.close_off_depth)
        self.age = f["age"][:]
        self.delta_age = np.ones_like(self.close_off_depth)
        self.d15n = f["d15N2"][:] - 1.
        #self.d15n_grav = f1['d15N2'][:] - 1.
        self.d15n_cod = np.ones_like(self.close_off_depth)
        self.d15n_grav_cod = np.ones_like(self.close_off_depth)
        self.d40ar = f["d40Ar"][:] - 1.
        #self.d40ar_grav = f1['d40Ar'][:] - 1.
        self.d40ar_cod = np.ones_like(self.close_off_depth)
        self.d40ar_grav_cod = np.ones_like(self.close_off_depth)
        #print(self.depth.shape[0])
        for i in range(self.depth.shape[0]):
            index = int(np.where(self.depth[i, 1:] == self.close_off_depth[i])[0])
            self.temperature_cod[i] = self.temperature[i, index]
            self.d15n_cod[i] = self.d15n[i, index]
            self.d40ar_cod[i] = self.d40ar[i, index]
            #self.d15n_grav_cod[i] = self.d15n_grav[i, index]
            #self.d40ar_grav_cod[i] = self.d40ar_grav[i, index]
            self.delta_age[i] = self.age[i, index] - self.close_off_age[i]
        self.delta_temp = self.climate[:, 2] - self.temperature_cod
        self.d15n_th_cod = self.d15n_cod - self.d15n_grav_cod
        self.d40ar_th_cod = self.d40ar_cod - self.d40ar_grav_cod

        f.close()
        return

    def plotting(self):
        fig, axs = plt.subplots(7, sharex=True, sharey=False)
        fig.set_figheight(15)
        fig.set_figwidth(8)
        # fig.suptitle('Sharing both axes')
        axs[0].plot(self.model_time, self.climate[:, 2], 'k-')
        axs[0].grid(linestyle='--', color='gray', lw='0.5')
        axs[0].set_ylabel(r"\centering Temperature\newline\centering Forcing [K]")
        axs[0].set_yticks(np.arange(230, 260, step=10))

        axs[1].plot(self.model_time, self.climate[:, 1], 'k-')
        axs[1].grid(linestyle='--', color='gray', lw='0.5')
        axs[1].set_ylabel(r"Accumulation\newline \centering Forcing\newline [$\mathrm{my}^{-1}$ ice eq.]")
        axs[1].set_yticks(np.arange(0.05, 0.6, step=0.2))

        axs[2].plot(self.model_time, self.close_off_depth, 'b-')
        axs[2].grid(linestyle='--', color='gray', lw='0.5')
        axs[2].set_ylabel(r"\centering Close-off\newline depth [m]")
        axs[2].set_yticks(np.arange(30, 120, step=30))
        axs[2].invert_yaxis()


        axs[3].plot(self.model_time, self.delta_temp, 'r-')
        axs[3].grid(linestyle='--', color='gray', lw='0.5')
        axs[3].set_ylabel(r"\centering Temperature \newline Gradient [K]")

        axs[4].plot(self.model_time, self.close_off_age, 'g-')
        axs[4].grid(linestyle='--', color='gray', lw='0.5')
        axs[4].set_ylabel(r"$\Delta$ age [y]")

        axs[5].plot(self.model_time, self.d15n_cod * 1000, 'c-', label='$\delta^{15}$N')
        axs[5].plot(self.model_time, self.d15n_th_cod * 1000, 'c:', label='$\delta^{15}$N thermal')
        #axs[5].plot(self.model_time, self.d15n_grav_cod * 1000, 'c--', label='$\delta^{15}$N gravitational')
        axs[5].grid(linestyle='--', color='gray', lw='0.5')
        axs[5].set_ylabel(r"\centering $\delta^{15}$N [‰]")
        axs[5].legend(loc='right', fontsize=8)
        axs[5].set_yticks(np.arange(0.0, 0.55, step=0.25))

        axs[6].plot(self.model_time, self.d40ar_cod / 4 * 1000, 'y-', label='$\delta^{40}$Ar')
        axs[6].plot(self.model_time, self.d40ar_th_cod / 4 * 1000, 'y:', label='$\delta^{40}$Ar thermal')
        #axs[6].plot(self.model_time, self.d40ar_grav_cod / 4 * 1000, 'y--', label='$\delta^{40}$Ar gravitational')
        axs[6].grid(linestyle='--', color='gray', lw='0.5')
        axs[6].set_ylabel(r"\centering $\delta^{40}$Ar/4 [‰]")
        axs[6].legend(loc='right', fontsize=8)
        axs[6].set_yticks(np.arange(0.0, 0.55, step=0.25))


        plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        #plt.savefig('resultsFolder/Test.pdf')
        plt.tight_layout
        plt.show()


        return


plots = CfmPlotter('CFM_2/CFM_main/CFMoutput_example/df/CFMresults.hdf5')
plots.plotting()
