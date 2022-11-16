from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py

# import a colormap
cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, 7)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class CfmPlotter():

    def __init__(self, f1path=None, f2path=None, f3path=None):
        path_list = [f1path, f2path, f3path]
        alpha = [1, 0.6, 0.3]
        label = ['Barnola', 'HLdynamic', 'Goujon']
        fig, axs = plt.subplots(8, sharex=True, sharey=False)
        fig.set_figheight(15)
        fig.set_figwidth(8)
        for k in range(len(path_list)):
            self.fpath = path_list[k]
            f = h5py.File(self.fpath)
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
            self.d15n_cod = np.ones_like(self.close_off_depth)
            self.d40ar = f["d40Ar"][:] - 1.
            self.d40ar_cod = np.ones_like(self.close_off_depth)
            self.gas_age = f["gas_age"][:]
            self.gas_age_cod = np.ones_like(self.close_off_depth)

            for i in range(self.depth.shape[0]):
                index = int(np.where(self.depth[i, 1:] == self.close_off_depth[i])[0])
                self.temperature_cod[i] = self.temperature[i, index]
                self.gas_age_cod[i] = self.gas_age[i, index]
                self.d15n_cod[i] = self.d15n[i, index]
                self.d40ar_cod[i] = self.d40ar[i, index]
                self.delta_age[i] = self.age[i, index] - self.gas_age[i, index]
            self.delta_temp = self.climate[:, 2] - self.temperature_cod

            # fig.suptitle('Sharing both axes')
            axs[0].plot(self.model_time, self.climate[:, 2], 'k-')
            axs[0].grid(linestyle='--', color='gray', lw='0.5')
            axs[0].set_ylabel(r"\centering Temperature\newline\centering Forcing [K]")
            axs[0].set_yticks(np.arange(230, 260, step=10))

            axs[1].plot(self.model_time, self.climate[:, 1], 'k-')
            axs[1].grid(linestyle='--', color='gray', lw='0.5')
            axs[1].set_ylabel(r"Accumulation\newline \centering Forcing\newline [$\mathrm{my}^{-1}$ ice eq.]")
            axs[1].set_yticks(np.arange(0.05, 0.6, step=0.2))

            axs[2].plot(self.model_time, self.close_off_depth, color=cmap(cmap_intervals[0]), alpha=alpha[k], label=label[k])
            axs[2].grid(linestyle='--', color='gray', lw='0.5')
            axs[2].set_ylabel(r"\centering Close-off\newline depth [m]")
            axs[2].set_yticks(np.arange(30, 120, step=30))
            axs[2].invert_yaxis()
            axs[2].legend(loc='lower right', fontsize=8)

            axs[3].plot(self.model_time, self.temperature_cod, color=cmap(cmap_intervals[1]), alpha=alpha[k], label=label[k])
            axs[3].grid(linestyle='--', color='gray', lw='0.5')
            axs[3].set_ylabel(r"\centering Temperature [K]")
            axs[3].legend(loc='lower right', fontsize=8)

            axs[4].plot(self.model_time, self.delta_temp, color=cmap(cmap_intervals[2]), alpha=alpha[k], label=label[k])
            axs[4].grid(linestyle='--', color='gray', lw='0.5')
            axs[4].set_ylabel(r"\centering Temperature \newline Gradient [K]")
            axs[4].legend(loc='lower right', fontsize=8)

            axs[5].plot(self.model_time, self.close_off_age, color=cmap(cmap_intervals[3]), alpha=alpha[k], label=label[k])
            axs[5].grid(linestyle='--', color='gray', lw='0.5')
            axs[5].set_ylabel(r"$\Delta$ age [y]")
            axs[5].legend(loc='lower right', fontsize=8)

            axs[6].plot(self.model_time, self.d15n_cod * 1000, color=cmap(cmap_intervals[4]), alpha=alpha[k], label=label[k])
            axs[6].grid(linestyle='--', color='gray', lw='0.5')
            axs[6].set_ylabel(r"\centering $\delta^{15}$N [‰]")
            axs[6].legend(loc='lower right', fontsize=8)
            axs[6].set_yticks(np.arange(0.0, 0.55, step=0.25))

            axs[7].plot(self.model_time, self.d40ar_cod / 4 * 1000, color=cmap(cmap_intervals[5]), alpha=alpha[k], label=label[k])
            axs[7].grid(linestyle='--', color='gray', lw='0.5')
            axs[7].set_ylabel(r"\centering $\delta^{40}$Ar/4 [‰]")
            axs[7].legend(loc='lower right', fontsize=8)
            axs[7].set_yticks(np.arange(0.0, 0.55, step=0.25))


        plt.xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        plt.savefig('results/periodic_T_5000y.pdf')
        plt.tight_layout
        plt.show()


        return


CfmPlotter('results/CFMresults_T_double_ramp_50y_500y_Barnola.hdf5',
            'results/CFMresults_T_double_ramp_50y_500y_HLdynamic.hdf5',
            'results/CFMresults_T_double_ramp_50y_500y_Goujon.hdf5'
           )

