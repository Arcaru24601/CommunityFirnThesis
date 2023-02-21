import h5py as hf

file = r'/home/jesperholm/Documents/CommunityFirnThesis-main/Python/Optimization/resultsFolder/Version1/HLdynamic/Dist_1/Point0.h5'
h5 = hf.File(file, 'r')
print(h5['count'][-1])
print(h5['d15N@CoD'][-1])
print(h5['temp'][-1])
print(h5['cost_func'][-1])
