# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:45:14 2019

@author: Abhishek Sharma
"""
value = X.isnull().sum()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def fill_missing_mean_data(data):
    return data.replace(np.NAN, data.mean())

def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(data)
    return label_encoder, encoded_categories

def categorical_missing_mean_data(data):
    label_encoder, encoded_category = encode_categorical_data(data.dropna())
    categorical_mean = int(round(encoded_category.mean()))
    return label_encoder.inverse_transform(categorical_mean)

dataset = pd.read_csv('train.csv')
X_columns = list(range(0, len(dataset.T)))
X_columns.remove(1)

X = dataset.iloc[:, X_columns]
y = dataset.iloc[:, 1]


# Data pre-processing
'''Extracting useful categorical features'''
categorical_features = []

for i in range (0, 11):
    if type(X.iloc[:, i].tolist()[1]) == str:
        categorical_features.append(i)

unwanted_categorical_features = \
    [data for data in categorical_features if len(X.iloc[:, data].unique())==1 \
     or len(X.iloc[:, data].unique()) > 15]
    
categorical_features = list(set(categorical_features) - 
                        set(unwanted_categorical_features))

'''Removing unwanted features from independent data'''
indepent_variable_cols = list(range(0,11))
classified_indepent_variable_cols = list(set(indepent_variable_cols )- 
                      set(unwanted_categorical_features))

'''Reintializing independent variable'''
X = X.iloc[:, classified_indepent_variable_cols]

'''Filling missing values'''
X['Age'] = fill_missing_mean_data(X['Age'])

X['Embarked'] = \
    X['Embarked'].replace(np.NAN, categorical_missing_mean_data(X['Embarked']))
    
'''Encoding categorical variables'''
gender_label_encoder, X['Sex'] = encode_categorical_data(X['Sex'])

embarked_label_encoder, X['Embarked'] = encode_categorical_data(X['Embarked'])

# Removing outliers from data
'''To be done latter'''

# Featureset selection using backward elemination
import statsmodels.formula.api as sm
X_feature_selection = np.append(arr=np.ones((891, 1)), values=X, axis=1)

X_opt = X_feature_selection[: , [0, 1, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_feature_selection[: , [0, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_feature_selection[: , [0, 2, 3, 4, 5, 6, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_feature_selection[: , [0, 2, 3, 4, 5, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

'''Creating new extracted independent variable'''

X_featured = X.iloc[:, [1,2,3,4,7]]

# Dividing data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_featured, y,
                                                    test_size=0.10, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting dta to logistics regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting test result
y_pred = classifier.predict(X_test)

# Creating Confusion matrix for accuracy understanding
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
