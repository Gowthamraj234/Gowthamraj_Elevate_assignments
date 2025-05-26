# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:32:24 2025

@author: GOWTHAMRAJ P R
"""
#1.importing dataset(titanic)

import os

import pandas as pd 

import numpy as np

df1 = pd.read_csv(r"D:\elevate labs\Titanic-Dataset.csv")

df1

type(df1)

#checking for null values

df1.info()

has_null = df1.isnull()

has_null

null_counts = df1.isnull().sum()

print(null_counts)

total_null_counts = df1.isnull().sum().sum()

print(total_null_counts)

#2.Handle missing values using mean/median/imputation

df1["Age"].fillna(df1["Age"].median(),inplace = True)

df1["Embarked"].replace(np.nan,df1["Embarked"].mode()[0],inplace = True)

df1["Cabin"].bfill(axis = "rows",inplace = True)

df1["Cabin"].ffill(axis = "rows",inplace = True)

df1.info()

#3.Convert categorical features into numerical using encoding

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoded = encoder.fit_transform(df1[["Sex"]]).toarray()

encoder_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out())

pd.concat([df1,encoder_df],axis=1)

from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()

lbl_encoded = lbl_encoder.fit_transform(df1[["Embarked"]])

#4.Normalize/standardize the numerical features.

#Normalization - minmax normalization

from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()

scaling.fit_transform(df1[["Age","Fare"]])

#Standardization(z score) normalization

from sklearn.preprocessing import StandardScaler

scaling1 = StandardScaler()

scaling1.fit_transform(df1[["Age","Fare"]])

#5.Visualizing and Removing Outliers Using Box Plot

import seaborn as sns

sns.boxplot(df1["Fare"])

sns.boxplot(df1["Age"])

df1["Age"].describe()

q1 = df1["Age"].quantile(0.25)

q3 = df1["Age"].quantile(0.75)

q1,q3

iqr = q3-q1

iqr

lower_limit =q1 - 1.5*iqr

upper_limit =q3 + 1.5*iqr

lower_limit,upper_limit

df1_no_outliers = df1[(df1["Age"]>lower_limit)&(df1["Age"]<upper_limit)]

sns.boxplot(df1_no_outliers["Age"])




