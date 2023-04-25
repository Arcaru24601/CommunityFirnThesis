# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:49:47 2023

@author: jespe
"""

import h5py as h5py
S = 800

cost_func = np.zeros(S)
d15N = np.zeros(S)
count = np.zeros(S)
Temp = np.zeros(S)

Temps = np.full(S,Input_temp[val])




Models = ['Ulti_Temp','Ulti_rho','Ulti_Deff']
for j in range(len(Models)):
    #j1 = Dists2[j]
   
    #Data_d15N = s[(abs(s - s.mean())) < (3 * s.std())][:S]

    print(Models[j])
    #for z,file in enumerate(glob.iglob('resultsFolder/Version1/' + str(Models[j]) + '/' + str(Dist[i]) + '/*.h5')):  
    for z in range(S):
        file = 'resultsFolder/' + str(Models[j]) + '/HLD/' +  + '5/Point' + str(z) + '.h5'
        #print(file)
        
        with h5py.File(file, 'r') as h5fr:
            #print(h5fr.keys())
            #print(2)
            cost_func[z] = h5fr['cost_func'][-1]
            d15N[z] = h5fr['d15N@CoD'][-1]
            count[z] = h5fr['count'][-1]
            Temp[z] = h5fr['temp'][-1]
            #print(np.mean(Temp))


Matrix = np.concatenate([count,Temps,Temp,d15N_dist,d15N], axis=1) 

df1 = pd.DataFrame(Matrix,
                   columns=['i', 'count', 'Ref Temp', 'Output temp', 'Ref d15N', 'Output d15N'])

df1.to_excel("output.xlsx")  
