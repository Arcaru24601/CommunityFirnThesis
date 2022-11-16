from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import h5py
import time
import os.path

# import a colormap
cmap = plt.cm.get_cmap('viridis')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class CfmPlotter():

    def __init__(self, fpath=None):
        # fpath = "./DO_results/DO_tests_vary_tr_time/cfm_DO_trtime_1500/Goujon_DO_trtime_1500.hdf5"
        self.d15N_cod = None
        self.fpath = fpath
        f = h5py.File(fpath)
        self.fs = os.path.split(fpath)[1]
        print(self.fs)
        print(f.keys())
        self.z = f["depth"][:]
        self.rho = f["density"][:]
        self.temperature = f["temperature"][:]
        self.age = f["age"][:]
        self.climate = f["Modelclimate"][:]
        self.model_time = np.array(([a[0] for a in self.z[:]]))
        self.close_off_depth = f["BCO"][:, 2]
        self.d15N = (f["d15N2"][:] - 1.) * 1000
        # get d15n at close-off depth (cod)
        self.d15N_cod = np.ones_like(self.model_time)
        for i in range(self.z.shape[0]):
            index = int(np.where(self.z[i, 1:] == self.close_off_depth[i])[0])
            self.d15N_cod[i] = self.d15N[i, index]
        f.close()
        return

    def init_plot(self):
        self.f0 = plt.figure(num=0, figsize=(11, 7), dpi=200)
        self.f0.tight_layout(pad=2.8)
        # self.f0.suptitle("CFM diffusion" , fontsize=12)
        self.ax01 = plt.subplot2grid((2, 3), (0, 1))
        self.ax02 = plt.subplot2grid((2, 3), (1, 1))
        self.ax03 = plt.subplot2grid((2, 3), (0, 0))
        self.ax13 = self.ax03.twinx()
        self.ax04 = plt.subplot2grid((2, 3), (1, 0))
        self.ax05 = plt.subplot2grid((2, 3), (0, 2))
        self.ax06 = plt.subplot2grid((2, 3), (1, 2))

        self.ax01.set_ylim((250, 0))
        self.ax02.set_ylim((250, 0))
        self.ax03.set_ylim((225, 250))
        self.ax02.set_xlim((225, 250))
        self.ax03.set_xlim((self.model_time[0], self.model_time[-1]))
        self.ax04.set_xlim((self.model_time[0], self.model_time[-1]))
        self.ax05.set_ylim((250, 0))
        self.ax06.set_ylim((130, 60))
        self.ax06.set_xlim((self.model_time[0], self.model_time[-1]))

        times = [500, 1500, 2000, 2500, 3000, 4000, 4990]
        cmap_intervals = np.linspace(0, 1, len(times))
        time_labels = ['$t_0$', '$t_1$', '$t_2$', '$t_3$', '$t_4$', '$t_5$', '$t_6$']

        self.ax01.set_ylabel(r"Depth [m]")
        self.ax01.set_xlabel(r"Density [$\mathrm{kgm}^{-3}$]", labelpad=-1.5, fontsize=9)  # labelpad=-1
        self.ax02.set_ylabel(r"Depth [m]")
        self.ax02.set_xlabel(r"Temperature [K]", labelpad=-1.5, fontsize=9)
        self.ax03.set_ylabel(r"Temperature Forcing [K]", color=cmap(cmap_intervals[0]))
        self.ax13.set_ylabel(r"Accumulation Forcing [$\mathrm{my}^{-1}$ ice eq.]", color=cmap(cmap_intervals[3]))
        self.ax03.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax04.set_ylabel(r"$\delta^{15}N$ [‰]", labelpad=-1.5, fontsize=9)
        self.ax04.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)
        self.ax05.set_ylabel(r"Depth [m]")
        self.ax05.set_xlabel(r"$\delta^{15}N$ [‰]", labelpad=-1.5, fontsize=9)
        self.ax06.set_ylabel(r"Close-off depth [m]")
        self.ax06.set_xlabel(r"Model Time [y]", labelpad=-1.5, fontsize=9)

        self.ax01.grid(linestyle=':', color='gray', lw='0.3')
        self.ax02.grid(linestyle=':', color='gray', lw='0.3')
        self.ax03.grid(linestyle=':', color='gray', lw='0.3')
        self.ax04.grid(linestyle=':', color='gray', lw='0.3')
        self.ax05.grid(linestyle=':', color='gray', lw='0.3')
        self.ax06.grid(linestyle=':', color='gray', lw='0.3')


        for i in range(len(times)):
            self.ax01.plot(self.rho[times[i]][1:], self.z[times[i]][1:], color=cmap(cmap_intervals[i]), linestyle='-', linewidth=0.7, label=time_labels[i])
            self.ax02.plot(self.temperature[times[i]][1:], self.z[times[i]][1:], color=cmap(cmap_intervals[i]), linewidth=0.7, label=time_labels[i])
            self.ax05.plot(self.d15N[times[i]][1:], self.z[times[i]][1:], color=cmap(cmap_intervals[i]), linewidth=0.7, label=time_labels[i])
        self.ax01.legend(loc='lower left', fontsize=8)
        self.ax02.legend(loc='lower left', fontsize=8)
        self.ax05.legend(loc='lower left', fontsize=8)

        self.ax03.plot(self.climate[:, 0], self.climate[:, 2], color=cmap(cmap_intervals[0]))
        self.ax13.plot(self.climate[:, 0], self.climate[:, 1], color=cmap(cmap_intervals[3]))

        self.ax03.axvline(x=-4500, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-4400, y=227, s='$t_0$')
        self.ax03.axvline(x=-3500, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-3400, y=227, s='$t_1$')
        self.ax03.axvline(x=-3000, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-2900, y=227, s='$t_2$')
        self.ax03.axvline(x=-2500, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-2400, y=227, s='$t_3$')
        self.ax03.axvline(x=-2000, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-1900, y=227, s='$t_4$')
        self.ax03.axvline(x=-1000, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-900, y=227, s='$t_5$')
        self.ax03.axvline(x=-30, linestyle='--', linewidth=0.5)
        self.ax03.text(x=-300, y=227, s='$t_6$')

        self.ax06.axvline(x=-4500, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-4400, y=127, s='$t_0$')
        self.ax06.axvline(x=-3500, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-3400, y=127, s='$t_1$')
        self.ax06.axvline(x=-3000, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-2900, y=127, s='$t_2$')
        self.ax06.axvline(x=-2500, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-2400, y=127, s='$t_3$')
        self.ax06.axvline(x=-2000, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-1900, y=127, s='$t_4$')
        self.ax06.axvline(x=-1000, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-900, y=127, s='$t_5$')
        self.ax06.axvline(x=-30, linestyle='--', linewidth=0.5)
        self.ax06.text(x=-300, y=127, s='$t_6$')

        self.ax04.plot(self.model_time, self.d15N_cod, color=cmap(cmap_intervals[2]), linewidth=0.7)
        self.ax06.plot(self.z[:, 0], self.close_off_depth, 'b-', color=cmap(cmap_intervals[2]), linewidth=0.7)

        plt.tight_layout(pad=2)
        # TODO: added plt.tight_layout; this helps a bit, however labels and headers still overlap
        plt.savefig('results/ramp_HLdynamic.pdf')
        plt.show()
        return

    def update_data(self, i):
        self.p011.set_data(self.rho[i][1:], self.z[i][1:])
        self.p012.set_data(self.temperature[i][1:], self.z[i][1:])
        self.p021.set_data(self.climate[:i, 0], self.climate[:i, 2])
        self.p022.set_data(self.climate[:i, 0], self.climate[:i, 1])
        self.p023.set_data(self.rho[i][1:], self.z[i][1:])
        # self.p024.set_data(self.rho[i][1:], self.z[i][1:])
        # self.iso_sigma18_co = np.append(self.iso_sigma18_co, self.iso_sigma18[i][1:][self.rho[i][1:] > 804.3][0])
        # self.p025.set_data(self.climate[:i, 0], self.iso_sigma18_co[:i])
        self.p025.set_data(self.z[:i, 0], self.close_off_depth[:i])
        # self.f0.suptitle(r"Model Phys: %s  Model time: %0.2f" % (self.fs, self.z[i][0]))
        # self.f0.suptitle('Model Phys ' + str(self.fs) + 'Model time: ' + str(self.z[i][0]))
        # TODO: include the modeltime and model physics... this really freaks me out
        self.f0.tight_layout(pad=2.8)
        return self.p011, self.p012, self.p021, self.p022, self.p023, self.p025  # ,self.p024,

    def plot_final(self):
        self.init_plot()
        self.p011, = self.ax01.plot(self.rho[0][1:], self.z[0][1:], 'b-')
        self.p012, = self.ax02.plot(self.temperature[0][1:], self.z[0][1:], 'k-')
        self.p021, = self.ax03.plot(self.climate[:, 0], self.climate[:, 2], 'k-')
        self.p022, = self.ax04.plot(self.climate[:, 0], self.climate[:, 1], 'k-')  #
        self.p023, = self.ax05.plot(self.rho[0][1:], self.z[0][1:], 'r-')
        # self.p024, = self.ax05.plot(self.rho[0][1:], self.z[0][1:], 'b-')
        # self.iso_sigmaD_co = np.array((self.iso_sigmaD[0][self.rho[0]>804.3][0],))
        # iso_sigma18_co = np.array(([self.iso_sigma18[i][1:][self.rho[i][1:] >= 804.3][0] for i in range(len(self.z))]))
        # self.iso_sigmaD_co = np.array(([self.iso_sigmaD[j][self.rho[j]>804.3][0] for j in np.arange(size(self.))]))
        # self.p025, = self.ax06.plot(self.climate[:, 0], iso_sigma18_co, 'b-')  #
        self.p025, = self.ax05.plot(self.z[:, 0], self.close_off_depth, 'b-')
        plt.show()
        return

    # def export_ascii(self, fout = None):
    #     """
    #     exports ascii with temp, accum history, close off depth and age as well as diffusion
    #     lengths at CO
    #     """
    #     model_time = np.array(([a[0] for a in self.z[:]]))
    #     temp_forcing = self.climate[:,2]
    #     accum_forcing = self.climate[:,1]
    #
    #     depth_co = np.array(([self.z[i][1:][self.rho[i][1:]>=804.3][0] for i in range(len(self.z))]))
    #     age_co = np.array(([self.age[i][1:][self.rho[i][1:]>=804.3][0] for i in range(len(self.z))]))
    #     sigma17_co = np.array(([self.iso_sigma17[i][1:][self.rho[i][1:]>=804.3][0] for i in range(len(self.z))]))
    #     sigma18_co = np.array(([self.iso_sigma18[i][1:][self.rho[i][1:]>=804.3][0] for i in range(len(self.z))]))
    #     sigmaD_co = np.array(([self.iso_sigmaD[i][1:][self.rho[i][1:]>=804.3][0] for i in range(len(self.z))]))
    #
    #     dataout = np.transpose(np.vstack((model_time, temp_forcing, accum_forcing, depth_co, age_co, sigma17_co,\
    #         sigma18_co, sigmaD_co)))
    #     dataout = np.delete(dataout, 0, 0)
    #
    #     if fout:
    #         f = open(fout, "w")
    #     else:
    #         f = open("./ramp_animation.out", "w")
    #     f.write("model_time\ttemp_forcing\tacuum_forcing\tdepth_co\tage_co\tsigma17_co\tsigma18_co\tsigmaD_co\n")
    #     np.savetxt(f, dataout, fmt = "%0.1f\t%0.2f\t%0.4e\t%0.3f\t%0.1f\t%0.6e\t%0.6e\t%0.6e")
    #     f.close()
    #
    #
    #     return


t1 = time.time()
# #plt.ion()
# folder_name = "./DO_results"
# list_of_files = os.listdir(folder_name)
#
# for j in list_of_files:
#     condition_1 = ("RC_tau" in j) & ("hdf5" in j) & ("Goujon" in j)  &\
#     ("acc300" in j)
#     if condition_1:
#         try:
#             t2 = time.time()
#             #Ascii out block
#             print("\n\nReading file %s" %folder_name + "/" + j)
#             plot_class = CfmPlotter(fpath = folder_name + "/" + j)
#             plot_class.export_ascii()
#             os.rename("./ramp_animation.out", "./" + os.path.splitext(j)[0] + ".out")
#             print("Saving file %s" %("./" + os.path.splitext(j)[0] + ".out"))
#
#             #Plotting block
#             plot_class.init_plot()
#             print len(plot_class.z)
#             anim = animation.FuncAnimation(plot_class.f0, plot_class.update_data, frames = np.int(len(plot_class.z)), interval = 10, blit = True)
#             anim.save('./basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#             os.rename("./basic_animation.mp4", "./" + os.path.splitext(j)[0] + ".mp4")
#             print("Saving file %s" %("./" + os.path.splitext(j)[0] + ".mp4"))
#             print("Loop time %0.1f minutes." %((time.time() - t2)/60.))
#
#
#         except IOError:
#             print("\nError reading %s " %folder_name + "/" + j)
#             continue
#     else:
#         print("Not processing file  %s" %folder_name + "/" + j)
#         continue


plt.close("all")

plot_class = CfmPlotter(fpath="resultsFolder/CFMresults_ramp_T_HLdynamic.hdf5")
# plot_class.export_ascii()

plot_class.init_plot()

# print len(plot_class.z)
# anim = animation.FuncAnimation(plot_class.f0, plot_class.update_data, frames=(len(plot_class.z)), interval=5, blit=True)

# anim.save('./basic_animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

plt.close("all")
# anim.save('./basic_animation.mp4', fps=50)


# plt.show()

print("Script time %0.1f minutes." % ((time.time() - t1) / 60.))
