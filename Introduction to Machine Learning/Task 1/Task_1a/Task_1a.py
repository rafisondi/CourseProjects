import numpy as np , pandas as pd
import csv
from math import sqrt

import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#init
lamda_ = [.01, .1 , 1 , 10, 100]
RMSE = np.zeros(5)
data  = pd.read_csv("train.csv").set_index("Id")
X = np.array(pd.DataFrame(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']]))
Y =data['y']
#train data
for lamda in lamda_:
    regr = lm.Ridge(alpha = lamda, fit_intercept=False)
    kf = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kf.split(X):
        regr.fit(X[train_index],Y[train_index])
        #predict values
        y_pred = regr.predict(X[test_index])
        RMSE[lamda_.index(lamda)] += mean_squared_error(Y[test_index], y_pred,squared = False) / 10.0
print(RMSE)
#save RMSE
wtr = csv.writer(open ('RMSE.csv', 'w'), delimiter=',', lineterminator='\n')
for x in RMSE :
     wtr.writerow ([x])
