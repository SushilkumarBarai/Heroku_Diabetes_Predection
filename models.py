# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:02:13 2020

@author: Sushilkumar
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:01:28 2020

@author: Sushilkumar
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier


##########################################SUV PRDECTION####################################################
#Reading CSV File
df=pd.read_csv("suv.csv")
#print(df)

x=df.iloc[:,[2,3]].values   #Taking Age and Salary column from csv file
y=df.iloc[:,4].values       #Taking Purchase from column csv file

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

model=LogisticRegression()
model.fit(x_train,y_train)

# Saving model # Saving model to file
pickle.dump(model, open('model_suv.pkl','wb'))
# Loading model to compare the results
model_suv = pickle.load(open('model_suv.pkl','rb'))

#############################################diabetes Predection###########################################################
#Reading CSV File diabetes

data = pd.read_csv("diabetes.csv")
data.shape
data.head(60)
# check if any null value is present or not
data.isnull().values.any()
data.corr()

#Changing the diabetes column data from boolean to number 0 or 1
diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)
data.head(5)
diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])
(diabetes_true_count,diabetes_false_count)

## Train Test Split data from csv file
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']

#print max and min value for a particular column for valiation in UI
print("num_preg",data.num_preg.min(),data.num_preg.max())
print("glucose_conc",data.glucose_conc.min(),data.glucose_conc.max())
print("diastolic_bp",data.diastolic_bp.min(),data.diastolic_bp.max())
print("insulin",data.insulin.min(),data.insulin.max())
print("bmi",data.bmi.min(),data.bmi.max())
print("diab_pred",data.diab_pred.min(),data.diab_pred.max())
print("age",data.age.min(),data.age.max())
print("skin",data.skin.min(),data.skin.max())

#SPLIT THE VALUE test and train
X = data[feature_columns].values
y = data[predicted_class].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

#Check how many other missing(zero) values
print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))

#Random Forest algorithm
random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train, y_train.ravel())

# Saving model to file
pickle.dump(random_forest_model, open('random_forest_model.pkl','wb'))
# Loading model to compare the results
random_forest_model = pickle.load(open('random_forest_model.pkl','rb'))

#Prediction testing
predict_train_data = random_forest_model.predict(X_test)
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

#Random cross checking
X_test=X_train[[535]]
X_test
print(X_test)
predict_train_data = random_forest_model.predict(X_test)
predict_train_data
print(predict_train_data)

#Diabetes Present
X_test=[[ 1.0  ,  88.0 ,   30.0 ,  99.0  ,   55.0 ,    1, 26.0 ,  1.6548]]
X_test
print(X_test)
predict_train_data = random_forest_model.predict(X_test)
predict_train_data
print(predict_train_data)
#print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

#diabetes Absent
X_test=X_train[[6]]
X_test
print(X_test)
predict_train_data = random_forest_model.predict(X_test)
predict_train_data
print(predict_train_data)
#print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))
