# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:25:53 2019

@author: Neel Roy
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