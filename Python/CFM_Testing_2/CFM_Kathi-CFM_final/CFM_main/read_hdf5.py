import h5py

def getKeysHdf5(filename):
    f = h5py.File(filename, "r")
    print("Keys: %s" % f.keys())
    return f.keys()

def getDataFromKey(filename, key):
    f = h5py.File(filename, "r")
    print(type(f[key]))
    return f[key]
