import numpy as np , pandas as pd
import csv
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import scale
from sklearn.impute import KNNImputer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GET DATA FROM TRAIN AND TEST FEATURES + ACCESS FUNCTION + IMPUTATION OF MISSING VALUES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#GET DATA FROM TRAIN AND TEST FEATURES
ttrain_features = pd.read_csv("train_features.csv") ###One patient: 12 rows 36 columns: ['pid', 'Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate','Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
ttest_features = pd.read_csv("test_features.csv")

id_keys_train = list(ttrain_features["pid"].drop_duplicates().to_numpy()) #18995 patients
id_keys_test = list(ttest_features["pid"].drop_duplicates().to_numpy()) #12664 patients
train_features.set_index('pid',inplace = True)

#Columns pid and Time are not relevant features and therefore removed
keep_col=['Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate','Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
train_features = ttrain_features[keep_col]
test_features = ttest_features[keep_col]
print(train_features)
print(test_features)

#ACCESS FUNCTION : INPUT: pid || OUTPUT: IF id is accepted --> Status of patient over last 24 hours [Pandas: DataFrame]
def get_patient_id(id):
    if id in list(id_keys_train):
        print("This is the patient data over the last 12 hours. \n")
        return train_features.groupby(train_features.index).get_group(id)
    else:
        print ("No valid PatientID")

print(train_features.mean())
print(test_features.mean())
#print(train_features.isnull().sum())

#SCALING AND IMPUTATION OF MISSING VALUES
x = np.arange(0,len(train_features),12)
for i in x:
    patient_data = train_features.iloc[i:i+12,:]
    imputer = KNNImputer(n_neighbors=1,copy=False)
    imputer.fit(patient_data)
    imputer.transform(patient_data)
    for j in range(len(keep_col)):
        if j==1:
            patient_data.iloc[:,j].fillna(40, inplace=True)
        elif j==2:
            patient_data.iloc[:,j].fillna(35, inplace=True)
        elif j==3:
            patient_data.iloc[:,j].fillna(13.5, inplace=True)
        elif j==4:
            patient_data.iloc[:,j].fillna(0.75, inplace=True)
        elif j==6:
            patient_data.iloc[:,j].fillna(14.5, inplace=True)
        elif j==8:
            patient_data.iloc[:,j].fillna(0, inplace=True)
        elif j==12:
            patient_data.iloc[:,j].fillna(8, inplace=True)
        elif j==13:
            patient_data.iloc[:,j].fillna(0.85, inplace=True)
        elif j==15:
            patient_data.iloc[:,j].fillna(30, inplace=True)
        elif j==16:
            patient_data.iloc[:,j].fillna(0.21, inplace=True)
        elif j==18:
            patient_data.iloc[:,j].fillna(97, inplace=True)
        elif j==19:
            patient_data.iloc[:,j].fillna(110, inplace=True)
        elif j==24:
            patient_data.iloc[:,j].fillna(9, inplace=True)
        elif j==27:
            patient_data.iloc[:,j].fillna(0.2, inplace=True)
        elif j==28:
            patient_data.iloc[:,j].fillna(101, inplace=True)
        elif j==29:
            patient_data.iloc[:,j].fillna(47, inplace=True)
        elif j==31:
            patient_data.iloc[:,j].fillna(0.8, inplace=True)
        elif j==32:
            patient_data.iloc[:,j].fillna(0.2, inplace=True)
        elif j==33:
            patient_data.iloc[:,j].fillna(-0.5, inplace=True)
    patient_data.fillna(train_features.mean(), inplace=True)
    if i%20000==0:
        print("giggity")

#add columns with similar values, so that we have fewer features
train_features.iloc[:,27] = train_features.iloc[:,27] + train_features.iloc[:,31] + train_features.iloc[:,32]
train_features.drop(columns=['Bilirubin_total', 'TroponinI'], inplace=True)

train_features.iloc[:,1] = train_features.iloc[:,1] + train_features.iloc[:,10]
train_features.drop(columns=['Fibrinogen'], inplace=True)

print("This is how the original training data after imputation looks like: ")
print(train_features)
print(train_features.iloc[0:12,:])


y = np.arange(0,len(test_features),12)
for i in y:
    patient_data = test_features.iloc[i:i+12,:]
    imputer = KNNImputer(n_neighbors=1,copy=False)
    imputer.fit(patient_data)
    imputer.transform(patient_data)
    for j in range(len(keep_col)):
        if j==1:
            patient_data.iloc[:,j].fillna(40, inplace=True)
        elif j==2:
            patient_data.iloc[:,j].fillna(35, inplace=True)
        elif j==3:
            patient_data.iloc[:,j].fillna(13.5, inplace=True)
        elif j==4:
            patient_data.iloc[:,j].fillna(0.75, inplace=True)
        elif j==6:
            patient_data.iloc[:,j].fillna(14.5, inplace=True)
        elif j==8:
            patient_data.iloc[:,j].fillna(0, inplace=True)
        elif j==12:
            patient_data.iloc[:,j].fillna(8, inplace=True)
        elif j==13:
            patient_data.iloc[:,j].fillna(0.85, inplace=True)
        elif j==15:
            patient_data.iloc[:,j].fillna(30, inplace=True)
        elif j==16:
            patient_data.iloc[:,j].fillna(0.21, inplace=True)
        elif j==18:
            patient_data.iloc[:,j].fillna(97, inplace=True)
        elif j==19:
            patient_data.iloc[:,j].fillna(110, inplace=True)
        elif j==24:
            patient_data.iloc[:,j].fillna(9, inplace=True)
        elif j==27:
            patient_data.iloc[:,j].fillna(0.2, inplace=True)
        elif j==28:
            patient_data.iloc[:,j].fillna(101, inplace=True)
        elif j==29:
            patient_data.iloc[:,j].fillna(47, inplace=True)
        elif j==31:
            patient_data.iloc[:,j].fillna(0.8, inplace=True)
        elif j==32:
            patient_data.iloc[:,j].fillna(0.2, inplace=True)
        elif j==33:
            patient_data.iloc[:,j].fillna(-0.5, inplace=True)
    patient_data.fillna(test_features.mean(), inplace=True)
    if i%20000==0:
        print("giggity")

#add columns with similar values, so that we have fewer features
test_features.iloc[:,27] = test_features.iloc[:,27] + test_features.iloc[:,31] + test_features.iloc[:,32]
test_features.drop(columns=['Bilirubin_total', 'TroponinI'], inplace=True)

test_features.iloc[:,1] = test_features.iloc[:,1] + test_features.iloc[:,10]
test_features.drop(columns=['Fibrinogen'], inplace=True)


print("This is how the original testing data after imputation looks like: ")
print(test_features)
print(test_features.iloc[0:12,:])

#scale all columns except for age
#for i in range(len(keep_col)-1):
#    scale(train_features.iloc[:,i+1], copy=False)
#    scale(test_features.iloc[:,i+1], copy=False)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#TRAINING AND PREDICTION
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Reshape training data so that we have one line per patient
print("One line per patient in training set")
train_data = np.reshape(train_features.to_numpy(),(18995,-1))
print(train_data)
print(len(train_data))

#Reshape testing data so that we have one line per patient
print("One line per patient in test set")
test_data = np.reshape(test_features.to_numpy(),(12664,-1))
print(test_data)
print(len(test_data))


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
