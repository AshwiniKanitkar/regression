#import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#if you need to install some packages, use this command:
#!pip install tensorflow
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#read the Dataset
salary_df = pd.read_csv('Salary_Data.csv')
salary_df.head(10)
salary_df.tail(10)


#define X variables and our target(y)
X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]
#splitting Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
#need to have y_train & y_test as an vector(for SageMaker linear learner) 
#y_train = y_train[:,0]
#y_test = y_test[:,0]
y_train = y_train.iloc[:,0].values
y_test = y_test.iloc[:,0].values

#pass in the training data from S3 to train the linear learner model
train = lin_reg.fit(X_train,y_train)
print("fit resyults", train)
print("Training done")
results = lin_reg.predict(X_test)
print("Prediction results",results)
print("Prediction done")
