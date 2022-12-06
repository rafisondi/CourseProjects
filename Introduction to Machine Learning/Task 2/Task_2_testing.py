import numpy as np , pandas as pd
import csv
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale
from sklearn.impute import KNNImputer


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GET DATA FOR TRAIN & TEST FEATURES       #GET DATA FOR TRAIN_LABELS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_features = pd.read_csv("train_features.csv")
train_features  = pd.DataFrame(train_features.drop(['Time', 'pid'], axis=1))          ### Training set: 1 patient 10 rows 36 columns: ['Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate','Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
test_features   = pd.read_csv("test_features.csv")                                         ### Testing set --- same setup as above

##########
data_train_label  = pd.read_csv("train_labels.csv").set_index("pid")                       ###Based on Features ---> Decide testing for <LABEL> [0,1]               [pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate]
                                                                                           ### ---[LABEL_TEST1 , LABEL_TEST2 (...) | LABEL_VITAL1 ,LABEL_VITAL2 , (...)]---

                                                                                           ### separate LABEL_TEST and LABEL_VITALS
train_tests = data_train_label.drop(['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'], axis=1)        ### Set of tests ordered --> Depending on i-th patient features
train_vitals = data_train_label[['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']]                      ### Measured i-th patient vitals

#print(train_features)

if __name__ == '__main__':



## Count of empty (NaN) values for each measured feature (Columns in Features)
    count_NaN_atFeature = []
    count = train_features.isna().sum()
    print(count.sort_values())
#    print(test_features.size())

#%%%%%%%%%%%%%%%%%%%%%%%%%
# TRAINING DATASET MODIFICATION:             Replace NaN with mean value of dataset & take peak value over 12h
#%%%%%%%%%%%%%%%%%%%%%%%%%

## train_features:      replace NaN values with %mean value% ( for now )

    train_features = train_features.fillna(train_features.mean())
    ## Deal with 12 hours of data --- take peak value over 12 hours
    train_features_max = pd.DataFrame(index = np.arange((len(train_features.index)-1)/12),columns = train_features.columns);
    #print(train_features_max)

 ###Get i-th patient 12h data
    print(np.arange(0,len(train_features),12))
    for i in np.arange(0,len(train_features),12) :
        patient_data = train_features.iloc[i:i+12,:]
        mean = train_features.mean()
        max = patient_data.max()
        min = patient_data.min()

        for j in range(len(max)):
            n = int(i/12)
            if abs(max[j]-mean[j]) < abs(min[j]-mean[j]):
                train_features_max.iat[n,j] = max[j]
            else:
                train_features_max.iat[n,j] = min[j]                                                                                   ### Save the highest values over 12h    --[panda(peak values features patient 1), pd(peak val patient 2), ...]--
    print("this is it")
    print(train_features_max)




"""
#%%%%%%%%%%%%%%%%%%%%%%%%%
# TESTING DATASET MODIFICATION:             Replace NaN with mean value of dataset & take peak value over 12h
#%%%%%%%%%%%%%%%%%%%%%%%%%

    test_features = test_features.fillna(test_features.mean())                                   ### Use mean value of training DATASET
## Deal with 12 hours of data --- take peak value over 12 hours
    train_max_features = []
    test_patient_data = 0
    x = np.arange(0,len(test_features),12)
    for i in x:
        test_patient_data = train_features.iloc[i:i+12,:]                                        ###Get i-th patient 12h data
        # reduce to one line
        train_max_features.append(test_patient_data.max())                                       ### Save the highest values over 12h    --[panda(peak values features patient 1), pd(peak val patient 2), ...]--
    #print(patient_features)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Classification
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#create empty dataframe where we are going to put the test results
labels = labels_of_tests + labels_of_vitals
test_labels = pd.DataFrame(index = id_keys_test, columns = labels)


#predicting whether tests should be carried out (classification)
for i in range(len(labels_of_tests)):
    clf = SVC(C=0.1, probability=True)
    clf.fit(train_data,np.array(tests.iloc[:,i]))
    print("Fitted")
    #test_labels.iloc[:,i] = 1 / (1+np.exp(-clf.decision_function(test_data)))
    test_labels.iloc[:,i] = np.array(clf.predict_proba(test_data))[:,1]
    #test_labels.iloc[:,i] = np.array(clf.predict(test_data))
    print( str(14-i) + " iterations to go" )

print("Results after classification:")
print(test_labels)

#predicting patient vitals (ridge regression with cross-validation)
for i in range(len(labels_of_vitals)):
    regr = RidgeCV(alphas=[1e-2, 1e-1, 1, 10, 100], fit_intercept=False)
    regr.fit(train_data ,vitals.iloc[:,i])
    test_labels.iloc[:,i+11] = regr.predict(test_data)
    print( str(3-i) + " iterations to go")

#Final result
print("Final result:")
print(test_labels)

#save test_labels
test_labels.to_csv('prediction.csv', index=True, index_label='pid', float_format='%.3f')
test_labels.to_csv('prediction.zip', index=True, index_label='pid', float_format='%.3f', compression='zip')

#create empty dataframe where we are going to put the test results
labels = labels_of_tests + labels_of_vitals
test_labels = pd.DataFrame(index = id_keys_test, columns = labels)

#predicting whether tests should be carried out (classification)
for i in range(len(labels_of_tests)):
    clf = SVC(C=0.1, probability=True)
    clf.fit(train_data,np.array(tests.iloc[:,i]))
    test_labels.iloc[:,i] = np.array(clf.predict_proba(test_data))[:,1]

"""
