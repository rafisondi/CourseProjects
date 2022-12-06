import numpy as np , pandas as pd
import csv
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale
from sklearn.impute import KNNImputer
import sys

#§§§§§§§§§§§§§§§§§§§§§FOR FUN%%%%%%%%%%%%%%%%%%%%%%%%
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    print("Done! \n")
    file.flush()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GET DATA FOR TRAIN & TEST FEATURES       #GET DATA FOR TRAIN_LABELS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_features = pd.read_csv("train_features.csv")
train_features  = pd.DataFrame(train_features.drop(['Time'], axis=1))       ### Training set: 1 patient 10 rows 37 columns: [pid , 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate','Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs','pH']

train_features = train_features.fillna(0)
train_features_reduced = pd.DataFrame()
for i in progressbar(range(0,len(train_features.index),12) , "Reduce Patient Features for TRAINING down to one line: ", 40):
    patient = train_features.iloc[i:i+12,:] #12 lines of patient data
    patient_red = patient.groupby('pid').aggregate(np.mean)
    train_features_reduced = pd.concat([train_features_reduced, patient_red],sort = False)
    #print(train_features_reduced)
train_features = train_features_reduced
#Reshape DataFrame to Numpy
train_data = train_features.to_numpy()



test_features   = pd.read_csv("test_features.csv")                                          ### Testing set --- same setup as above
test_features  = pd.DataFrame(test_features.drop(['Time'], axis=1))
test_features = test_features.fillna(0)
test_features_reduced = pd.DataFrame()
for i in progressbar(range(0,len(test_features.index),12) , "Reduce Patient Features for TESTING down to one line: ", 40):
    patient = test_features.iloc[i:i+12,:] #12 lines of patient data
    patient_red = patient.groupby('pid').aggregate(np.mean)
    test_features_reduced = pd.concat([test_features_reduced, patient_red],sort = False)
test_features = test_features_reduced
#Reshape DataFrame to Numpy
test_data = test_features.to_numpy()

##########
data_train_label  = pd.read_csv("train_labels.csv").set_index("pid")                       ###Based on Features ---> Decide testing for <LABEL> [0,1]               [pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate]
                                                                                           ### ---[LABEL_TEST1 , LABEL_TEST2 (...) | LABEL_VITAL1 ,LABEL_VITAL2 , (...)]---

                                                                                           ### separate LABEL_TEST and LABEL_VITALS

train_tests = data_train_label.drop(['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'], axis=1)        ### Set of tests ordered --> Depending on i-th patient features
train_vitals = data_train_label[['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']]                    ### Measured i-th patient vitals

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#FIT MDOEL TO DATA
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Empty DataFrame for TESTING results
test_labels = pd.DataFrame(index = test_features.index, columns = data_train_label.columns)
#print(test_labels.shape)

for i in progressbar(range(len(train_tests.columns)), "Fitting model to reduced patient data: ", 40):
    clf = SVC(C=0.1, probability=True)
    clf.fit(train_data , np.array(data_train_label.iloc[:,i]))
    test_labels.iloc[:,i] = np.array(clf.predict_proba(test_data))[:,1]

for i in progressbar(range(len(train_vitals.columns)) , "Ridge regression fit of patients vitals: ", 40):
    regr = RidgeCV(alphas=[1e-2, 1e-1, 1, 10, 100], fit_intercept=False)
    regr.fit(train_data ,train_vitals.iloc[:,i])
    test_labels.iloc[:,i+11] = regr.predict(test_data)

test_labels.to_csv('prediction.csv', index=True, index_label='pid', float_format='%.3f')
test_labels.to_csv('prediction.zip', index=True, index_label='pid', float_format='%.3f', compression='zip')
