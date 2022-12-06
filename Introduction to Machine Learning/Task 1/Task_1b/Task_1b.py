import numpy as np , pandas as pd
import csv
import math

import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

data  = pd.read_csv("train.csv").set_index("Id")
X = np.array(pd.DataFrame(data[['x1', 'x2', 'x3', 'x4', 'x5']]))
Y =data['y']

phi = []
w_minimized = []
#w_aver = []
num_of_folds = 100


def phi_f(X):
    phi = np.zeros(( np.size(X,0) , 21 ))
    for i in range(np.size(X,0)):
        for j in range(21):
            if       j <5   :  phi[i,j] = X[i,(j%5)]
            if 5  <= j <10  :  phi[i,j] = X[i,(j%5)]**2
            if 10 <= j <15  :  phi[i,j] = math.exp(X[i,(j%5)])
            if 15 <= j <20  :  phi[i,j] = np.cos(X[i,(j%5)])
            if 20 <= j      :  phi[i,j] = 1
    return phi

#print(pd.DataFrame(data[['x1', 'x2', 'x3', 'x4', 'x5']]))
regr = lm.Ridge(alpha = 0, fit_intercept=False)
kf = KFold(n_splits=num_of_folds, shuffle=False)
for train_index, test_index in kf.split(X):
    regr.fit(phi_f(X[train_index]) ,Y[train_index])
    y_pred = regr.predict(phi_f(X[test_index]))
    w_minimized.append(regr.coef_)
    #error += mean_squared_error(Y[test_index], y_pred,squared = False) / num_of_folds
print(pd.DataFrame(w_minimized))

#Get Average w_minimized
w_aver = pd.DataFrame(w_minimized).mean(axis = 0, skipna = True)
print(w_aver)

#regr.fit(phi_f(X) , Y)
#w_minimized = regr.coef_
#print(w_minimized)

#save w_minimized
wtr = csv.writer(open ('w_minimized.csv', 'w'), delimiter=',', lineterminator='\n')
for x in np.array(w_aver) :
     wtr.writerow ([x])
