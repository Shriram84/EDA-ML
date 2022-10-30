#!/usr/bin/env python
# coding: utf-8

# 1 Problem Statement
# • To predict the temperature using Algerian forest fire dataset
# 

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)


# In[77]:


url = "https://raw.githubusercontent.com/subhashdixit/Linear_Regression/main/Algerian_Forest_Dataset/Algerian_forest_fires_dataset.csv"
df=pd.read_csv(url,header=1 )


# In[78]:


df


# Data Checks and cleaning

# Checking Null Values

# In[79]:


df[df.isnull().any(axis=1)]


# Dropping the row with null value

# In[80]:


df.drop([122,123,167],axis=0, inplace=True)
df = df.reset_index()
df.head()


# Show all the columns

# In[81]:


df.columns


# Checking Columns with Extra Spaces

# In[82]:


[x for x in df.columns if ' ' in x]


# Remove extra space in column names
# 

# In[83]:


df.columns = df.columns.str.strip()
df.columns


# In[84]:


import re
def Remove_Extra_Space(x):
    return (re.sub(' +', ' ', x).strip())


# Remove extra space in the data

# In[85]:


df['Classes'] = df['Classes'].apply(Remove_Extra_Space)


# Drop extra index column

# In[86]:


df.drop(['index'],axis=1, inplace=True)


# Create date feature with the help of day, month and year feature and convert to datetime

# In[87]:


df['date'] = pd.to_datetime(df[['day', 'month', 'year']])


# Drop day, month and year feature

# In[88]:


df.drop(['day', 'month', 'year'], axis = 1, inplace = True)


# Imputation of date based on temperature. Usually in summer temperature is more and in winter it is less. So, we have categorized it based on month

# In[89]:


def date_imputation(x):
  if (x >= pd.to_datetime('2012-07-01')) and (x <= pd.to_datetime('2012-09-01')):
    return 1
  else:
    return 0
df['date'] = df['date'].apply(date_imputation)


# In[90]:


df['date'].value_counts()


# Sidi-Bel Abbes Region and Bejaia Region - are classified with 0 and 1

# In[91]:


df.loc[:122, 'Region'] = 0
df.loc[122:, 'Region'] = 1


# check null values in all the features

# In[92]:


df.isnull().sum()


# Mapping Classes as 1 & 0 for No fire, fire

# In[93]:


df['Classes'] = df['Classes'].map({'not fire' : 0, 'fire': 1})


# Check duplictes values in all the column

# In[94]:


df.duplicated().sum()


# Check data types of all the features

# In[95]:


df.dtypes


# Convert features to its logical datatypes

# In[96]:


convert_data = {'Temperature' : 'float64', 'RH': 'float64', 'Ws': 'float64',
 'DMC' : 'float64', 'DC' : 'float64', 'ISI': 'float64', 'BUI': 'float64', 'FWI' : 'float64', 
 'Region' : 'object', 'Rain' : 'float64', 'FFMC' : 'float64' , 'Classes':'object','date':'object'}
df = df.astype(convert_data)


# In[97]:


df.dtypes #converted


# Unique values

# In[98]:


df.nunique()


# Statics of Data Set

# In[99]:


df.describe()


# Categorical features

# In[100]:


categorical_feature=[feature for feature in df.columns if df[feature].dtypes=='O']
categorical_feature


# In[101]:


for feature in categorical_feature:
 print(df.groupby(feature)[feature].value_counts())


# Numerical Features

# In[102]:


numerical_features=[feature for feature in df.columns if df[feature].dtypes!='O']
print(numerical_features)


# Discrete feature from Numerical Feature

# In[103]:


#the assumption to consider a feature discrete is that it should have less than 35 unique values otherwise it will be 
# considered continuous feature
discrete_features=[feature for feature in numerical_features if len(df[feature].unique())<18]
discrete_features


# Continuous Features

# In[104]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_features]
print(continuous_features)


# # Graphical Analysis

# In[105]:


#observing distribution for continuous feature
for feature in continuous_features:
    plt.figure(figsize=(12,8))
    sns.histplot(data=df, x=feature,kde=True, bins=30, color='blue')
    plt.show();


# # Outliers Handling

# In[106]:


#Prior to removing outliers
plt.figure(figsize=(20, 8))
sns.boxplot(data=df)
plt.title("Before Removing Outliers")


# In[107]:


#Upper & Lower boundaries
def find_boundaries(df, variable, distance):
 IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
 lower_boundary = df[variable].quantile(0.25) - (IQR*distance)
 upper_boundary = df[variable].quantile(0.75) + (IQR*distance)
 return upper_boundary, lower_boundary


# In[108]:


#outlier Deletion
outliers_columns = ['Temperature', 'Ws','Rain','FFMC','DMC','ISI','BUI', 'FWI']
for i in outliers_columns:
  upper_boundary, lower_boundary = find_boundaries(df,i, 1.5)
  outliers = np.where(df[i] > upper_boundary, True, np.where(df[i] < lower_boundary, True, False))
  outliers_df = df.loc[outliers, i]
  df_trimed= df.loc[~outliers, i]
  df[i] = df_trimed


# In[109]:


#After removing outliers
plt.figure(figsize=(15, 8))
sns.boxplot(data=df)
plt.title("After Removing Outliers")


# In[110]:


df.isnull().sum() #null value check


# In[113]:


df.fillna(df.median().round(1), inplace=True) #Imputation of null values from features


# In[114]:


df.isnull().sum()


# # Statistical Analysis

# Correlation of numerical variable

# In[115]:


data = round(df.corr(),2)


# In[112]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(data=data, annot=True,  vmin=-1, vmax=1)


# DMC and BUI are highly correlated 0.97, will drop BUI

# In[116]:


df.drop('BUI', axis=1, inplace=True)


# # Model Building

# Independent features vs target features distribution

# In[117]:


sns.scatterplot(data=df, x='date', y='Temperature', hue='Classes' )


# In[118]:


df.columns


# In[119]:


df.info()


# # Regression Plot

# In[47]:


for feature in [feature for feature in df.columns if feature not in['Temperature', 'date', 'Region', 'Classes']]:
    sns.set(rc={'figure.figsize':(8,8)})
    sns.regplot(x=df[feature], y=df['Temperature'])
    plt.xlabel(feature)
    plt.ylabel("Temperature")
    plt.title("{} Vs Temperature".format(feature))
    plt.show();


# Seperating dependent and independent feature
# 

# In[120]:


X= df[['RH', 'Ws', 'Rain','FFMC', 'DMC', 'ISI','DC',
'FWI', 'Classes', 'Region', 'date']]
y=df[['Temperature']]


# In[121]:


X.head()


# In[122]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42) #training and test data bifurcation


# In[123]:


### creating a StandardScalar object
scaler=StandardScaler()
scaler


# In[124]:


### Using fit_transform to standardise Train data
X_train=scaler.fit_transform(X_train)


# In[125]:


X_test=scaler.transform(X_test)


# In[126]:


X_train


# # Linear Regression Model

# In[127]:


## creating linear regression model
linear_reg=LinearRegression()
linear_reg


# In[128]:


pd.DataFrame(X_train).isnull().sum()


# In[129]:


### Passing training data(X and y) to the model
linear_reg.fit(X_train, y_train)


# In[130]:


### Printing co-efficients and intercept of best fit hyperplane
print("1. Co-efficients of independent features is {}".format(linear_reg.coef_))
print("2. Intercept of best fit hyper plane is {}".format(linear_reg.intercept_))


# Test Data Prediction

# In[131]:


linear_reg_pred=linear_reg.predict(X_test)


# In[132]:


residual_linear_reg=y_test-linear_reg_pred
residual_linear_reg = pd.DataFrame(residual_linear_reg)


# Validation of Linear Regression assumptions

# In[133]:


plt.scatter(x=y_test,y=linear_reg_pred)
plt.xlabel("Test truth data")
plt.ylabel("Predicted data")


# We can observe the linear relation between Predicted and test truth data - Validated

# In[134]:


sns.displot(data=residual_linear_reg, kind='kde')


# Residual is normally distributed - Validated

# In[135]:


plt.scatter(x=linear_reg_pred, y=residual_linear_reg)
plt.xlabel('Predictions')
plt.ylabel('Residuals')


# Residual and Predicted values should follow uniform distribution - Observed and Validated

# Cost Function Values

# In[139]:


print(f"MSE : {round(mean_squared_error(y_test, linear_reg_pred),2)}\nMAE :{round(mean_absolute_error(y_test, linear_reg_pred),2)}\nRMSE : {round(np.sqrt(mean_squared_error(y_test, linear_reg_pred)),2)}")


# Performance Metrics

# In[140]:


linear_reg_r2_score=r2_score(y_test, linear_reg_pred)
linear_reg_adj_r2_score=1-((1-linear_reg_r2_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print(f"R-Squared Accuracy : {round(linear_reg_r2_score*100,3)} % \nAdjusted R-Squared Accuracy : {round(linear_reg_adj_r2_score*100,2)}%")


# # Ridge Regresion Model

# In[142]:


ridge_reg=Ridge()
ridge_reg


# In[143]:


ridge_reg.fit(X_train, y_train)


# In[144]:


### Printing co-efficients and intercept of best fit hyperplane
print("1. Co-efficients of independent features is {}".format(ridge_reg.coef_))
print("2. Intercept of best fit hyper plane is {}".format(ridge_reg.intercept_))


# Test Data Prediction

# In[145]:


ridge_reg_pred=ridge_reg.predict(X_test)


# In[146]:


residual_ridge_reg=y_test-ridge_reg_pred
residual_ridge_reg = pd.DataFrame(residual_ridge_reg)


# Validation of Ridge Regression assumptions
# 

# In[147]:


plt.scatter(x=y_test,y=ridge_reg_pred)
plt.xlabel("Test truth data")
plt.ylabel("Predicted data")


# Linear Relationship observed between Predicted data & Test Truth Data - Validated

# In[148]:


sns.displot(data = residual_ridge_reg, kind='kde')


# Residual is normally distributed - Validated

# In[149]:


plt.scatter(x=ridge_reg_pred, y=residual_ridge_reg)
plt.xlabel('Predictions')
plt.ylabel('Residuals')


# There is a uniform distribution when Residulas plotted against Predictions - Validated

# Cost Function Values:

# In[151]:


print(f"MSE : {round(mean_squared_error(y_test, ridge_reg_pred),2)}\nMAE :{round(mean_absolute_error(y_test, ridge_reg_pred),2)}\nRMSE : {round(np.sqrt(mean_squared_error(y_test, ridge_reg_pred)),2)}")


# Performance Metrics

# In[152]:


ridge_reg_r2_score=r2_score(y_test, ridge_reg_pred)
ridge_reg_adj_r2_score=1-((1-ridge_reg_r2_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print(f"R-Squared Accuracy : {round(ridge_reg_r2_score*100,3)} % \nAdjusted R-Squared Accuracy : {round(ridge_reg_adj_r2_score*100,2)}%")


# # Lasso Regression Model

# In[153]:


lasso_reg=Lasso()
lasso_reg


# In[154]:


## Passing training data(X and y) to the model
lasso_reg.fit(X_train, y_train)


# In[155]:


## Printing co-efficients and intercept of best fit hyperplane
print("1. Co-efficients of independent features is {}".format(lasso_reg.coef_))
print("2. Intercept of best fit hyper plane is {}".format(lasso_reg.intercept_))


# In[156]:


lasso_reg_pred=lasso_reg.predict(X_test) #prediction of test data


# In[157]:


y_test = y_test.squeeze()
residual_lasso_reg = y_test-lasso_reg_pred
residual_lasso_reg = pd.DataFrame(residual_lasso_reg)


# # Assumtion Validation - Lasso

# In[158]:


plt.scatter(x=y_test,y=lasso_reg_pred)
plt.xlabel("Test truth data")
plt.ylabel("Predicted data")


# Linear Relation observed between predicted data and test data - Validated

# In[159]:


sns.displot( data = residual_lasso_reg, kind='kde')


# Residual is following a normal(Guassian) distribution when plotted Kernel Density Function - Validated

# In[160]:


plt.scatter(x=lasso_reg_pred, y=residual_lasso_reg)
plt.xlabel('Predictions')
plt.ylabel('Residuals')


# Uniform distribution observed between residual and predictions - Validated

#  Cost Function Values

# In[161]:


print(f"MSE : {round(mean_squared_error(y_test, lasso_reg_pred),2)}\nMAE :{round(mean_absolute_error(y_test, lasso_reg_pred),2)}\nRMSE : {round(np.sqrt(mean_squared_error(y_test, lasso_reg_pred)),2)}")


# Performance Metrics

# In[164]:


lasso_reg_r2_score=r2_score(y_test, lasso_reg_pred)
lasso_reg_adj_r2_score=1-((1-lasso_reg_r2_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print(f"R-Squared Accuracy : {round(lasso_reg_r2_score*100,3)} % \nAdjusted R-Squared Accuracy : {round(lasso_reg_adj_r2_score*100,2)}%")


# In[ ]:




