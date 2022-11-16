import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

time_exp1 = np.loadtxt('time/time_exp1.txt')
time_spdiags = np.loadtxt('time/time_spdiags.txt')
time_where1 = np.loadtxt('time/time_where1.txt')
time_where2 = np.loadtxt('time/time_where2.txt')
time_where3 = np.loadtxt('time/time_where3.txt')
time_interp1 = np.loadtxt('time/time_interp1.txt')
time_interp2 = np.loadtxt('time/time_interp2.txt')
time_interp3 = np.loadtxt('time/time_interp3.txt')
time_intersect = np.loadtxt('time/time_intersect1.txt')
time_spsolve = np.loadtxt('time/time_spsolve.txt')
time_sum = np.loadtxt('time/time_sum.txt')
time_T = np.loadtxt('time/time_T.txt')
time_T2 = np.loadtxt('time/time_T2.txt')
time_reshape1 = np.loadtxt('time/time_reshape1.txt')
time_reshape2 = np.loadtxt('time/time_reshape2_log.txt')

# list_times = [time_spdiags, time_spsolve, time_intersect, time_T, time_T2, time_sum, time_reshape1, time_reshape2]
# list_legend = ['spdiags', 'spsolve', 'intersect1', 'T1', 'T2', 'sum', 'reshape1', 'reshape2']

list_times = [time_T, time_sum, time_reshape1, time_reshape2]
list_legend = ['T1', 'sum', 'reshape1', 'reshape2']

cmap = plt.cm.get_cmap('viridis')
cmap_intervals = np.linspace(0, 1, len(list_times) + 1)

for i in range(len(list_times)):
    plt.plot(list_times[i]/10**6, color=cmap(cmap_intervals[i]), label=list_legend[i])
plt.grid(linestyle='--', color='gray', lw='0.5')
plt.xlabel('Number of calls')
plt.ylabel('Time per call [ms]')
plt.legend()
plt.show()