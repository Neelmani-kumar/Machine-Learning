# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:55:06 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import seaborn as sb 
import statsmodels.api as sm
from sklearn import datasets

# Importing the datasets

data = pd.read_csv("C:/Users/Neel Roy/Desktop/boston_data.csv",encoding='unicode_escape') #read the dat file
print(data.keys()) # Print name of variables
print("Data Shape:", data.shape) #print no of rows & columns
print(data.head()) #print top 5 observations
print(data.dtypes) #print data type of variables

# matplotlib histogram
plt.hist(data['medv'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
# Density Plot and Histogram 
sb.distplot(data['medv'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

#Scatter Plots
#sb.pairplot(data,kind='reg')

# Log Transform
#data['logmedv']=np.log(data['medv'])

#Outlier Detection
#Univariate Approach
sb.boxplot(x=data['crim'])
sb.boxplot(x=data['zn'])
sb.boxplot(x=data['indus'])
sb.boxplot(x=data['nox'])
sb.boxplot(x=data['rm'])
sb.boxplot(x=data['age'])
sb.boxplot(x=data['dis'])
sb.boxplot(x=data['rad'])
sb.boxplot(x=data['tax'])
sb.boxplot(x=data['ptratio'])
sb.boxplot(x=data['black'])
sb.boxplot(x=data['lstat'])
sb.boxplot(x=data['medv'])

#IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
p=((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR)))

#RemovingOutliers
#data_out = data[((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
#data_out.shape

#Outliers Treatment
UL=(Q3 + 1.5 * IQR)

mask = data.crim > 10.00932
data.loc[mask,'crim'] = 10.00932

mask = data.zn > 31.25000
data.loc[mask,'zn'] = 31.25000

mask = data.rm >  7.71900
data.loc[mask,'rm'] =  7.71900

mask = data.dis >  9.92350
data.loc[mask,'dis'] =  9.92350

mask = data.lstat > 31.57250
data.loc[mask,'lstat'] =  31.57250

mask = data.medv > 36.85000
data.loc[mask,'medv'] =  36.85000

#Missing Value Detection 
data.info()
data.isnull().sum()

#Missing Value Treartment
#mean_value=data['zn'].mean()
#data['zn']=data['zn'].fillna(mean_value)
#this will replace all NaN values with the mean of the non null values


#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit(data)
#train= imp.transform(data)
#This will look for all columns where we have NaN value and replace
#the NaN value with specified test statistic.


#Correlation
corr=data.corr()

#Correlation Heat Map(Optional)
fig = plt.subplots(figsize = (10,10))
sb.set(font_scale=1.5)
sb.heatmap(data.corr(),square = True,cbar=True,annot=True,
           annot_kws={'size': 10})
plt.show()

# Declare Dependent variable & create independent & dependent datasets
dep = "medv"
X = data.drop(dep,axis=1)
Y = data[dep]

# with statsmodels
X = sm.add_constant(X) # adding a constant

#Split data into train & test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,
        Y,test_size=0.2,random_state=5) #random_state is the seed used by the random number generator;

#Model Building
lm=sm.OLS(Y_train, X_train).fit()
lm.summary()

lm1=sm.OLS(Y_train, X_train.drop(['zn','indus','chas','age'], 
                                 axis=1)).fit()
lm1.summary()

lm2=sm.OLS(Y_train, X_train.drop(['zn','indus','chas','nox','crim',
                                  'rm','rad','age','tax','lstat',
                                  'ptratio'], axis=1)).fit()
lm2.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
x_train=X_train.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat'], axis=1)
[variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])]

#Prediction
pred_test=lm2.predict(X_test.drop(['zn','indus','chas','nox','crim','rm','rad','age','tax','lstat','ptratio'],axis=1))
err_test=np.abs(Y_test - pred_test)
print(err_test)

#MAPE
import numpy as np

def mean_absolute_percentage_error(Y_test, pred_test): 
    Y_test, pred_test = np.array(Y_test), np.array(pred_test)
    return np.mean(np.abs((Y_test - pred_test) / Y_test)) * 100

mean_absolute_percentage_error(Y_test, pred_test)

#Linearity
#The Null hypothesis is that the regression is correctly modeled as linear
sm.stats.diagnostic.linear_harvey_collier(lm2)
#Plot
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(err_test, pred_test)

#Heterosckedasticity
#null hypothesis is that all observations have the same error variance, 
#i.e. errors are homoscedastic.
_, pval, __, f_pval = sm.stats.diagnostic.het_breuschpagan(err_test,X_test[['const','dis', 'black']])
print( pval, f_pval)

#Normality
import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(err_test, plot=ax, fit=True)
r**2

from scipy import stats
stats.kstest(err_test, 'norm')

#Box Cox Transformation
#from scipy.stats import boxcox
#boxcox_transformed_data = boxcox(X_train['dis'])
#X_train.loc[:,14] = boxcox_transformed_data[0]

# Log Transform
#data['logmedv']=np.log(data['medv'])
#sb.distplot(data['logmedv'], hist=True, kde=True, 
#          hist_kws={'edgecolor':'black'},
#          kde_kws={'linewidth': 4})