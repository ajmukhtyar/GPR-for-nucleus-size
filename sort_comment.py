# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:32:04 2020

@author: ankitamukhtyar
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 



lowerlim = 4
upperlim = 50


PE0 = [[] for i in range(lowerlim,upperlim)]
PE_scaled = [[] for i in range(lowerlim,upperlim)]
TE0 = [[] for i in range(lowerlim,upperlim)]
TE_scaled = [[] for i in range(lowerlim,upperlim)]
vol0 = [[] for i in range(lowerlim,upperlim)]
vol_scaled = [[] for i in range(lowerlim,upperlim)]
num = [[] for i in range(lowerlim,upperlim)]


########### Reading in the data ###########

for i in range(lowerlim,upperlim):
    print(i)
    with open("input_files/output%s" %i) as f:
        save = 0 
        for line in f:
            if line.startswith('Step'):
                # beginning of section use first lin
                for line in f:
                    # check for end of section breaking if we find the stop lone
                    if line.startswith("Loop"):
                        break
                    else: 
                        if len(PE0[i-lowerlim])==0:
                            PE0[i-lowerlim].append(float(line.split()[2]))
                            TE0[i-lowerlim].append(float(line.split()[4]))
                            vol0[i-lowerlim].append(float(line.split()[6]))
                        elif float((line.split)()[0]) != save:
                            PE0[i-lowerlim].append(float(line.split()[2]))
                            TE0[i-lowerlim].append(float(line.split()[4]))
                            vol0[i-lowerlim].append(float(line.split()[6]))
                        save = float(line.split()[0])
        
        #------------Normalizing the data------------------------------
        data=list(zip(PE0[i-lowerlim],vol0[i-lowerlim],TE0[i-lowerlim]))
        scaled_data = scaler.fit_transform(data)
        PE_scaled[i-lowerlim]=(scaled_data[:,0]) 
        vol_scaled[i-lowerlim]=(scaled_data[:,1]) 
        TE_scaled[i-lowerlim]=scaled_data[:,2]


    with open("input_files/totord%s.dat" %i) as f2:
        count=0
        for line1 in f2:
            if count%100==0:
                num[i-lowerlim].append(float(line1.split()[0]))
            count+=10

#---------------------------------------------

PE = PE_scaled
vol = vol_scaled
TE = TE_scaled
listPE = np.concatenate((PE))
listvol = np.concatenate((vol))
listTE = np.concatenate((TE))
Y = np.concatenate((num))

nucl=[0,1,2,3,4,5,6,7,8]
peavg=[]
volavg=[]
teavg= []
nums=[] 

#------- In each of the simulations finding the average potential energy/total energy/volume for a given nucleus size ----------

for k in range(len(PE)):
    for j in nucl:
        pe2=[]
        vol2=[]
        te2=[]
        for i in range(len(PE[k])):
            if num[k][i]==j:
                pe2.append(PE[k][i])
                vol2.append(vol[k][i])
                te2.append(TE[k][i])
        if math.isnan(np.mean(pe2)) ==False and math.isnan(np.mean(vol2)) ==False and math.isnan(np.mean(te2)) ==False: 
            peavg.append(np.mean(pe2))
            volavg.append(np.mean(vol2))
            teavg.append(np.mean(te2))
            nums.append(j)

#---------------- Writing the data file to be used as input for the gaussian process regression -----------------------------
''' First 5 columns are the features we want to use an input 
1: potential energy
2: volume
3: total energy
4: pe + vol
5: pe * vol

The last two features are to see if a linear or multiplicative combination of the inputs improves results

Column 6 has the nucleus size label 
'''

f2 = open("data_normalized.txt","w")
for i in range(len(peavg)):
    f2.write(str(peavg[i])+"\t"+str(volavg[i])+"\t"+str(teavg[i])+"\t"+str(peavg[i]+volavg[i])+"\t"+str(peavg[i]*volavg[i])+"\t"+str(nums[i])+"\n")

f2.close()



