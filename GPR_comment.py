# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:30:38 2020

@author: ankitamukhtyar
"""

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt


#----------- Reading data into matrix --------------------
f = open("data_normalized.txt")

c=0
for line in f:
   c=c+1

Xdata = np.zeros([c,5])
Y = np.zeros(c)


f1 = open("data_normalized.txt")
i=0
for line1 in f1:
    Xdata[i][0] = float(line1.split()[0])
    Xdata[i][1] = float(line1.split()[1])
    Xdata[i][2] = float(line1.split()[2])
    Xdata[i][3] = float(line1.split()[3])
    Xdata[i][4] = float(line1.split()[4])

    Y[i] = float(line1.split()[5])
    i+=1
#----------------------------------------------

#---------- create training and testing variables and fit GPR model ----------------
X_train, X_test, y_train, y_test = train_test_split(Xdata, Y, test_size=0.2)

kernel =  gp.kernels.ConstantKernel(10.0, (1e-1, 1e3))* gp.kernels.RBF(10.0, (1e-3, 1e3)) 

model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
model.fit(X_train, y_train)
params = model.kernel_.get_params()
y_pred, std = model.predict(X_test, return_std=True)
MSE = ((y_pred-y_test)**2).mean()
print((MSE))

#------ Rounding of the predictions since nucleus size has to be an integer -----------
ypredint=[np.round(j) for j in y_pred]

#-------- Plotting result ----------
''' We want the predicted values for nucleus size y_pred to be as close as possible to y_actual '''

plt.plot(y_test,y_pred,'ro')
plt.plot(y_test,y_test,color='blue',label="Expected")
plt.legend(frameon=False)
plt.xlabel('Actual')
plt.ylabel("Prediction")
plt.show()

