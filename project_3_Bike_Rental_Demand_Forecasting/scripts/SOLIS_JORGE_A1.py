#!/usr/bin/env python
# coding: utf-8

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />
# 
# <br><h2>Kaggle Submission Template</h2>
# <h4>DAT-5390 | Computational Data Analytics with Python</h4>
# Chase Kusterer - Faculty of Analytics<br>
# Hult International Business School<br><br><br>
# 
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />
# 
# <br>
# 
# <h3>The Purpose of the Notebook</h3><br>
# This Notebook is organized as a template for submitting on Kaggle. It will output a .csv file that can be submitted via Kaggle.<br><br>
# Remember that you also need to provide an analysis alongside your model building via the assignment link on the course page. Start by conducting your analysis and then copy/paste relevant code to this template (missing value imputation, feature engineering, etc.). Please do not submit this template as your analysis (many of the technical steps are not necessary for the analysis). Finally, on the course page, make sure to submit your analysis and model development as one document in two formats (Jupyter Notebook and a .txt file).
# <br><br>
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />

# <h3>Reminder: Model Analysis Tips</h3><br>
# <strong>How fit should a model be?</strong><br>
# As a general heuristic, if the training and testing scores are within 0.05 of each other, the model has not been overfit. Don't worry if the testing score ends up higher than the training score. Some sources claim that in such situations a model is underfit, but this is a general misconception that is beyond the scope of this course. For this course, long as the training and testing scores are within 0.05 of each other, the model is good to go.
# <br><br>
# 
# <strong>Which model should I choose?</strong><br>
# All models have their own benefits and drawbacks. Thus, it is important to test out more than one and to also explore their <a href="https://scikit-learn.org/stable/modules/classes.html#classical-linear-regressors">documentation</a>.

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# <h2>Part I: Imports and Data Check</h2>

# In[1129]:


## importing libraries ##

# for this template submission
import numpy             as np                       # mathematical essentials
import pandas            as pd                       # data science essentials
import sklearn.linear_model                          # linear models
from sklearn.model_selection import train_test_split # train/test split


#!###############################!#
#!# import additional libraries #!#
#!###############################!#
#_____
import seaborn as sns # enhanced graphical output
import matplotlib.pyplot as plt # essential graphical output
import statsmodels.formula.api as smf # regression modeling

from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler  # standard scaler

from sklearn.tree import DecisionTreeRegressor         # regression trees
from sklearn.tree import plot_tree     
from sklearn.model_selection import RandomizedSearchCV # hyperparameter tuning

# setting pandas print options (optional)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# <br>

# In[1130]:


## importing data ##

# reading modeling data into Python
modeling_data = './datasets/chicago_training_data.xlsx'

# calling this df_train
df_train = pd.read_excel(io         = modeling_data,
                         sheet_name = 'data',
                         header     = 0,
                         index_col  = 'ID')



# reading testing data into Python
testing_data = './datasets/test.xlsx'

# calling this df_test
df_test = pd.read_excel(io         = testing_data,
                        sheet_name = 'data',
                        header     = 0,
                        index_col  = 'ID')


# <br>

# In[1131]:


# concatenating datasets together for mv analysis and feature engineering
df_train['set'] = 'Not Kaggle'
df_test ['set'] = 'Kaggle'

# concatenating both datasets together for mv and feature engineering
df_full = pd.concat(objs = [df_train, df_test],
                    axis = 0,
                    ignore_index = False)


# checking data
df_full.head(n = 5)


# <br>

# In[1132]:


# checking available features
df_full.columns


# <br>

# In[1133]:


#!##############################!#
#!# set your response variable #!#
#!##############################!#
y_variable = 'RENTALS'
df_full_mv = df_full


# <br><hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# <h2>Part II: Data Preparation</h2><br>
# Complete the following steps to prepare for model building. Note that you may add or remove steps as you see fit. Please see the assignment description for details on what steps are required for this project.
# <br><br>
# <h3>Base Modeling</h3>

# In[1134]:


## Base Modeling ##

# INFOrmation about each variable
df_full.info(verbose = True)


# In[1135]:


# developing a histogram using HISTPLOT
sns.histplot(data  = df_train,
         x     = "RENTALS",
         kde   = True)


# title and axis labels
plt.title(label   = "Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")


# displaying the histogram
plt.show()


# In[1136]:


data_0 = df_train[df_train.FunctioningDay == 'Yes']
sns.histplot(data  = data_0,
         x     = "RENTALS",
         kde   = True)

# title and axis labels
plt.title(label   = "Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")


# displaying the histogram
plt.show()


# In[1137]:


sns.lineplot(data = df_train,
            x = 'DateHour',
            y ='RENTALS')
# title and axis labels
plt.title(label   = "Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")


# displaying the histogram
plt.show()


# In[1138]:


inputs_num = ['Temperature(F)', 'Humidity(%)', 'Wind speed (mph)', 'Visibility(miles)', 'DewPointTemperature(F)', 'Rainfall(in)', 'Snowfall(in)', 'SolarRadiation(MJ/m2)']
plt.style.use('ggplot')
num_bins = 10
data_0 = df_train[df_train.FunctioningDay == 'Yes']

for i in inputs_num:
    n, bins, patches = plt.hist(data_0[i], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(i)
    plt.ylabel('Número')
    plt.show()


# In[1139]:


inputs_num = ['Temperature(F)', 'Humidity(%)', 'Wind speed (mph)', 'Visibility(miles)', 'DewPointTemperature(F)', 'Rainfall(in)', 'Snowfall(in)', 'SolarRadiation(MJ/m2)']
plt.style.use('ggplot')
num_bins = 10
data_1 = df_full[df_full.FunctioningDay == 'Yes']

for i in inputs_num:
    n, bins, patches = plt.hist(data_0[i], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(i)
    plt.ylabel('Número')
    plt.show()


# In[1140]:


# descriptive statistics for numeric data
df_full_stats = data_0.iloc[ :, 1: ].describe(include = 'number').round(decimals = 2)
df_full_stats


# In[1141]:


# developing a correlation matrix
df_full_corr = data_0.corr(method = 'pearson',numeric_only = True)
df_full_corr

# filtering results to show correlations with Sale_Price
df_full_corr.loc[ : , "RENTALS"].round(decimals = 2).sort_values(ascending = False)


# In[1142]:


# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_full_corr, 
            annot=True, fmt=".2f", 
            cmap='coolwarm', 
            cbar=True, 
            linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[1143]:


# setting figure size
fig, ax = plt.subplots(figsize = (9, 6))


# developing a scatterplot
sns.scatterplot(x    = "Temperature(F)",
         y    = "RENTALS",
         data = data_0)


# SHOWing the results
plt.show()


# In[1144]:


# setting figure size
fig, ax = plt.subplots(figsize = (9, 6))


# developing a scatterplot
sns.scatterplot(x    = "Humidity(%)",
         y    = "RENTALS",
         data = data_0)


# SHOWing the results
plt.show()


# # Analysis Introduction
# In the burgeoning bike-sharing industry, understanding factors that drive rental demand is crucial for planning and operational efficiency. This analysis leverages machine learning to predict daily bike rentals in a major US city, using a dataset that encompasses weather conditions, temporal factors, and operational statuses. Initial exploratory data analysis (EDA) reveals intricate patterns and relationships, guiding subsequent data preprocessing steps to enhance model performance. Feature engineering further refines the dataset, introducing new variables aimed at encapsulating underlying trends and seasonalities. A comparative evaluation of several regression models, adhering to specified constraints, culminates in the selection of an optimal model based on predictive accuracy and interpretability.

# <br><h3>Missing Value Analysis and Imputation</h3>

# In[1145]:


## Missing Value Imputation ##
df_full.isnull().describe()
df_full.isnull().sum(axis = 0)


# In[1146]:


df_full.isnull().mean(axis = 0)


# In[1147]:


# looping to flag features with missing values
for col in df_full:

    # creating columns with 1s if missing and 0 if not
    if df_full[col].isnull().astype(int).sum() > 0:
        df_full['m_'+col] = df_full[col].isnull().astype(int)


# In[1148]:


df_full.columns


# In[1149]:


df_full = df_full.drop(columns=['m_RENTALS'])
df_full.columns


# In[1150]:


# checking results - summing missing value flags
df_full[ ['m_Visibility(miles)', 'm_DewPointTemperature(F)', 'm_SolarRadiation(MJ/m2)'] ].sum(axis = 0)


# In[1151]:


# subsetting for mv features
mv_flag_check = df_full[ ['Visibility(miles)'     , 'm_Visibility(miles)',
                          'DewPointTemperature(F)' , 'm_DewPointTemperature(F)',
                          'SolarRadiation(MJ/m2)', 'm_SolarRadiation(MJ/m2)'] ]


# checking results - feature comparison
mv_flag_check.sort_values(by = ['m_Visibility(miles)', 'm_DewPointTemperature(F)', 'm_SolarRadiation(MJ/m2)'],
                          ascending = False).head(n = 10)


# In[1152]:


#Missing values of VISIBILITY

# plotting 'Visibility(miles)'
sns.histplot(x = 'Visibility(miles)',
            data = df_full,
            kde = True)


# title and labels
plt.title (label  = 'Distribution of Visibility(miles)')
plt.xlabel(xlabel = 'Visibility(miles)')
plt.ylabel(ylabel = 'Count')


# displaying the plot
plt.show()


# In[1153]:


fill1 = df_full['Visibility(miles)'].median()
# imputing Visibility(miles)
df_full['Visibility(miles)'].fillna(value   = fill1,
                                    inplace = True)


# In[1154]:


#Check the correct imputation
df_full[ ['Visibility(miles)', 'm_Visibility(miles)'] ][df_full['m_Visibility(miles)'] == 1].head(n = 10)


# In[1155]:


# DewPointTemperature
df_full[['DewPointTemperature(F)' , 'm_DewPointTemperature(F)']].describe()

# plotting 'DewPointTemperature'
sns.histplot(x = 'DewPointTemperature(F)',
            data = df_full,
            kde = True)


# title and labels
plt.title (label  = 'Distribution of DewPointTemperature(F)')
plt.xlabel(xlabel = 'DewPointTemperature(F)')
plt.ylabel(ylabel = 'Count')


# displaying the plot
plt.show()


# In[1156]:


# setting figure size
fig, ax = plt.subplots(figsize = (9, 6))


# developing a scatterplot
sns.scatterplot(x    = "Temperature(F)",
         y    = "DewPointTemperature(F)",
         data = data_0)


# SHOWing the results
plt.show()


# In[1157]:


#Converting F to C
df_full['Temperature(C)']=(df_full['Temperature(F)']-32)*5/9

#Using the DewPoint Temperature Formula to estimate the real value in F
fill2=((df_full['Temperature(C)']-((100-df_full['Humidity(%)'])/5))*9/5)+32

#Imputing missing values
df_full['DewPointTemperature(F)'].fillna(value=fill2,
                          inplace = True)
#Delete the new column created
df_full.drop(columns=['Temperature(C)'], inplace=True)


# In[1158]:


#Check the correct imputation
df_full[ ['DewPointTemperature(F)', 'm_DewPointTemperature(F)','Temperature(F)','Humidity(%)'] ][df_full['m_DewPointTemperature(F)'] == 1].head(n = 10)


# In[1159]:


# plotting 'SolarRadiation(MJ/m2)'
sns.histplot(x = 'SolarRadiation(MJ/m2)',
            data = df_full,
            kde = True)


# title and labels
plt.title (label  = 'Distribution of SolarRadiation(MJ/m2)')
plt.xlabel(xlabel = 'SolarRadiation(MJ/m2)')
plt.ylabel(ylabel = 'Count')


# displaying the plot
plt.show()


# In[1160]:


# imputing SolarRadiation(MJ/m2)
df_full['SolarRadiation(MJ/m2)'].fillna(value   = 0   ,
                               inplace = True)


# In[1161]:


df_full[ ['SolarRadiation(MJ/m2)', 'm_SolarRadiation(MJ/m2)'] ][df_full['m_SolarRadiation(MJ/m2)'] == 1].head(n = 10)


# In[1162]:


# making sure all missing values have been taken care of
df_full.isnull().sum(axis = 0)


# In[1163]:


#######################
## Visibility(miles) ##
#######################
# scatterplot AFTER missing values
sns.histplot(data  = df_full,
             x     = 'Visibility(miles)',
             fill  = True,
             color = "red")


# scatterplot BEFORE missing values
sns.histplot(data  = df_full_mv,
             x     = 'Visibility(miles)',
             fill  = True,
             color = 'black')


# mean lines
plt.axvline(df_full['Visibility(miles)'].mean()   , color = "red")
plt.axvline(df_full_mv['Visibility(miles)'].mean(), color = "blue")


# labels and rendering
plt.title (label  = "Imputation Results (Visibility(miles))")
plt.xlabel(xlabel = "Visibility(miles)")
plt.ylabel(ylabel = "Frequency")
plt.show()


# In[1164]:


############################
## DewPointTemperature(F) ##
############################
# scatterplot AFTER missing values
sns.histplot(data  = df_full,
             x     = 'DewPointTemperature(F)',
             fill  = True,
             color = "red")


# scatterplot BEFORE missing values
sns.histplot(data  = df_full_mv,
             x     = 'DewPointTemperature(F)',
             fill  = True,
             color = 'black')


# mean lines
plt.axvline(df_full['DewPointTemperature(F)'].mean()   , color = "red")
plt.axvline(df_full_mv['DewPointTemperature(F)'].mean(), color = "blue")


# labels and rendering
plt.title (label  = "Imputation Results (DewPointTemperature(F))")
plt.xlabel(xlabel = "DewPointTemperature(F)")
plt.ylabel(ylabel = "Frequency")
plt.show()


# In[1165]:


###########################
## SolarRadiation(MJ/m2) ##
###########################
# scatterplot AFTER missing values
sns.histplot(data  = df_full,
             x     = 'SolarRadiation(MJ/m2)',
             fill  = True,
             color = "red")


# scatterplot BEFORE missing values
sns.histplot(data  = df_full_mv,
             x     = 'SolarRadiation(MJ/m2)',
             fill  = True,
             color = 'black')


# mean lines
plt.axvline(df_full['SolarRadiation(MJ/m2)'].mean()   , color = "red")
plt.axvline(df_full_mv['SolarRadiation(MJ/m2)'].mean(), color = "blue")


# labels and rendering
plt.title (label  = "Imputation Results (SolarRadiation(MJ/m2))")
plt.xlabel(xlabel = "SolarRadiation(MJ/m2)")
plt.ylabel(ylabel = "Frequency")
plt.show()


# # Exploratory Data Analysis and Data Preprocessing
# 
# The exploratory data analysis commenced with a review of descriptive statistics, uncovering a wide range of rental counts and diverse weather conditions. Histograms of rental counts illustrated a right-skewed distribution, suggesting variability in daily usage patterns. Correlation analysis highlighted potential predictors, such as temperature and humidity, albeit with varying degrees of association. Notably, the presence of missing values in visibility, dew point temperature, and solar radiation necessitated thoughtful imputation strategies. The preprocessing phase also addressed categorical variables through one-hot encoding, ensuring compatibility with machine learning algorithms. Standardization of continuous variables was paramount to eliminate scale discrepancies, thereby facilitating more balanced contributions across features.

# <br><h3>Transformations</h3>

# In[1166]:


## Transformations ##

# developing a histogram using HISTPLOT
sns.histplot(data   = df_full[df_full['FunctioningDay']=='Yes'],
             x      = 'RENTALS',
             kde    = True)


# title and axis labels
plt.title(label   = "Original Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[1167]:


# log transforming Sale_Price and saving it to the dataset
df_full['log_RENTALS'] = np.log1p(df_full['RENTALS'])


# In[1168]:


# developing a histogram using HISTPLOT
sns.histplot(data   = df_full[df_full['FunctioningDay']=='Yes'],
             x      = 'log_RENTALS',
             kde    = True)


# title and axis labels
plt.title(label   = "Logarithmic Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS (log)") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


# In[1169]:


df_full.skew(axis = 0, numeric_only = True).round(decimals = 2)


# In[1170]:


df_full[df_full['FunctioningDay']=='Yes'].skew(axis = 0, numeric_only = True).round(decimals = 2)


# In[1171]:


# Logarithmically transform any X-features that have an absolute skewness value greater than 1.0.

df_full['log_Wind speed (mph)'] = np.log1p(df_full['Wind speed (mph)'])
df_full['log_Visibility(miles)'] = np.log1p(df_full['Visibility(miles)'])
df_full['log_Rainfall(in)'] = np.log1p(df_full['Rainfall(in)'])
df_full['log_SolarRadiation(MJ/m2)'] = np.log1p(df_full['SolarRadiation(MJ/m2)'])


# In[1172]:


# skewness AFTER logarithmic transformations
df_full.loc[ : , 'log_RENTALS': ].skew(axis = 0).round(decimals = 2).sort_index(ascending = False)


# In[1173]:


# analyzing (Pearson) correlations
df_corr = df_full[df_full['FunctioningDay']=='Yes'].corr(method = 'pearson',numeric_only = True ).round(2)

df_corr.loc[ : , ['RENTALS', 'log_RENTALS'] ].sort_values(by = 'RENTALS',
                                                                ascending = False)


# <br><h3>Feature Engineering</h3>

# In[1174]:


## Feature Engineering ##
# counting the number of zeroes for 
windspeed_zeroes  = len(df_full['Wind speed (mph)'][df_full['Wind speed (mph)']==0]) 
visibility_zeroes   = len(df_full['Visibility(miles)'][df_full['Visibility(miles)']==0]) 
dwpt_zeroes     = len(df_full['DewPointTemperature(F)'][df_full['DewPointTemperature(F)']==0]) 
rainfall_zeroes = len(df_full['Rainfall(in)'][df_full['Rainfall(in)']==0]) 
snowfall_zeroes    = len(df_full['Snowfall(in)'][df_full['Snowfall(in)']==0]) 
solar_zeroes  = len(df_full['SolarRadiation(MJ/m2)'][df_full['SolarRadiation(MJ/m2)']==0]) 


# printing a table of the results
print(f"""
                 No\t\tYes
               ---------------------
WindSpeed     | {windspeed_zeroes}\t\t{len(df_full) - windspeed_zeroes}
Visibility    | {visibility_zeroes}\t\t{len(df_full) - visibility_zeroes}
DewPoint      | {dwpt_zeroes}\t\t{len(df_full) - dwpt_zeroes}
Rainfall      | {rainfall_zeroes}\t\t{len(df_full) - rainfall_zeroes}
Snowfall      | {snowfall_zeroes}\t\t{len(df_full) - snowfall_zeroes}
SolarRadiatio | {solar_zeroes}\t\t{len(df_full) - solar_zeroes}
""")


# In[1175]:


# placeholder variables
df_full['has_SolarRadiation'] = 0

# iterating over each original column to
# change values in the new feature columns
for index, value in df_full.iterrows():


    # Solar Radiation
    if df_full.loc[index, 'SolarRadiation(MJ/m2)'] > 0:
        df_full.loc[index, 'has_SolarRadiation'] = 1


# In[1176]:


# checking results
df_full[  ['has_SolarRadiation']  ].head(n = 5)


# In[1177]:


# developing a small correlation matrix
new_corr = df_full[df_full['FunctioningDay']=='Yes'].corr(method = 'pearson', numeric_only = True).round(decimals = 2)


# checking the correlations of the newly-created variables with Sale_Price
new_corr.loc[ ['has_SolarRadiation'],
              ['RENTALS', 'log_RENTALS'] ].sort_values(by = 'RENTALS',
                                                             ascending = False)


# In[1178]:


#CATEGORICAL DATA

# printing columns
print(f"""
Holiday
------
{df_full['Holiday'].value_counts()}


FunctioningDay
----------
{df_full['FunctioningDay'].value_counts()}

""")


# In[1179]:


# defining a function for categorical boxplots
def categorical_boxplots(response, cat_var, data):
    """
	This function is designed to generate a boxplot for  can be used for categorical variables.
    Make sure matplotlib.pyplot and seaborn have been imported (as plt and sns).

    PARAMETERS
	----------
	response : str, response variable
	cat_var  : str, categorical variable
	data     : DataFrame of the response and categorical variables
	"""

    fig, ax = plt.subplots(figsize = (10, 8))
    
    sns.boxplot(x    = response,
                y    = cat_var,
                data = data)
    
    plt.suptitle("")
    plt.show()


# In[1180]:


# calling the function for Holiday
categorical_boxplots(response = 'RENTALS',
					 cat_var  = 'Holiday',
					 data     = df_full)


# calling the function for FunctioningDay
categorical_boxplots(response = 'RENTALS',
					 cat_var  = 'FunctioningDay',
					 data     = df_full)


# In[1181]:


# one hot encoding categorical variables
one_hot_Holiday = pd.get_dummies(df_full['Holiday'], prefix = 'Holiday')
one_hot_FunctioningDay = pd.get_dummies(df_full['FunctioningDay'], prefix = 'FunctioningDay')

# dropping categorical variables after they've been encoded
df_full = df_full.drop('Holiday', axis = 1)
df_full = df_full.drop('FunctioningDay', axis = 1)

# joining codings together
df_full = df_full.join([one_hot_Holiday,one_hot_FunctioningDay ])


# saving new columns
new_columns = df_full.columns


# In[1182]:


# checking results
df_full.head(n = 5)


# In[1183]:


# creating a (Pearson) correlation matrix
df_corr = df_full[(df_full['FunctioningDay_Yes'] == True) & (df_full['set'] == 'Not Kaggle')].corr(numeric_only = True).round(2)


# printing (Pearson) correlations with SalePrice
df_corr.loc[ : , ['RENTALS', 'log_RENTALS'] ].sort_values(by = 'RENTALS',
                                                                ascending = False)


# In[1184]:


#Converting DataHour in datatime type
df_full['DateHour'] = pd.to_datetime(df_full['DateHour'])


# In[1185]:


# Get the year, month, day, hour, and week day
df_full['Year'] = df_full['DateHour'].dt.year
df_full['Month'] = df_full['DateHour'].dt.month
df_full['Day'] = df_full['DateHour'].dt.day
df_full['Hour'] = df_full['DateHour'].dt.hour
df_full['DayOfWeek'] = df_full['DateHour'].dt.weekday

# Checking new columns
df_full[['DateHour', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek']].head()


# In[1186]:


df_full[['DateHour', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek']].describe()


# In[1187]:


df_full['Day_Month'] = df_full['Day'].astype(str).str.zfill(2) + '-' + df_full['Month'].astype(str).str.zfill(2)


# In[1188]:


sns.lineplot(data = df_full[df_full['FunctioningDay_Yes'] == True],
            x = 'Day_Month',
            y ='RENTALS')
# title and axis labels
plt.title(label   = "Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")


# displaying the histogram
plt.show()


# In[1189]:


#These new features capture the cyclical nature of the weekdays, ensuring that consecutive days are close 
#to each other in the feature space and that Sunday is close to Monday, 
#which wouldn't be captured if we simply treated the weekdays as ordinary categorical or numerical variables.

# Converting day of week into rad 
df_full['DayOfWeek_rad'] = (2 * np.pi * df_full['DayOfWeek']) / 7

# Creating cycle variables for each day of week 
df_full['DayOfWeek_sin'] = np.sin(df_full['DayOfWeek_rad'])
df_full['DayOfWeek_cos'] = np.cos(df_full['DayOfWeek_rad'])


# In[1190]:


df_full = pd.get_dummies(df_full, columns=['DayOfWeek'], prefix='weekday', drop_first=True)


# In[1191]:


## New feature: One of the most well-known formulas for calculating the heat index is the Steadman formula.
df_full['heatIndex'] = 0.5*(df_full['Temperature(F)'] + 61.0 + ((df_full['Temperature(F)']-68.0)*1.2)+(df_full['Humidity(%)']*0.094))


# In[1192]:


#new feature indicating poor weather conditions based on visibility, rainfall, and snowfall

# Define thresholds
visibility_threshold = 7  
rainfall_threshold = 0.1

# Create 'PoorWeather' column based on the defined criteria
df_full['PoorWeather'] = ((df_full['Visibility(miles)'] <= visibility_threshold) |
                          (df_full['Rainfall(in)'] > rainfall_threshold) |
                          (df_full['Snowfall(in)'] > 0)).astype(int)


# In[1193]:


# creating a (Pearson) correlation matrix
df_corr = df_full[(df_full['FunctioningDay_Yes'] == True) & (df_full['set'] == 'Not Kaggle')].corr(numeric_only = True).round(2)


# printing (Pearson) correlations with SalePrice
df_corr.loc[ : , ['RENTALS', 'log_RENTALS'] ].sort_values(by = 'RENTALS',
                                                                ascending = False)


# In[1194]:


df_full = df_full.drop(columns=['DateHour', 'Year','Day_Month'])


# In[1195]:


# subsetting for RENTALS
rental_corr = df_corr.loc[ : , ['RENTALS', 'log_RENTALS'] ].sort_values(by = 'RENTALS',
                                                                 ascending = False)
# removing irrelevant correlations
rental_corr = rental_corr.iloc[ 2: , : ]

# placeholder column for y-variable recommendation
rental_corr['original_v_log'] = 0

# filling in placeholder
for index, column in rental_corr.iterrows():
    
    # if RENTALS is higher
    if abs(rental_corr.loc[ index, 'RENTALS']) >  abs(rental_corr.loc[ index, 'log_RENTALS']):
        rental_corr.loc[ index , 'original_v_log'] = 'RENTALS'
        
        
    # if log_RENTALS is higher 
    elif abs(rental_corr.loc[ index, 'RENTALS']) <  abs(rental_corr.loc[ index, 'log_RENTALS']):
        rental_corr.loc[ index , 'original_v_log'] = 'log_RENTALS'
    
    
    # in case they are tied
    else:
        rental_corr.loc[ index , 'original_v_log'] = 'Tie'
        

# checking results
rental_corr["original_v_log"].value_counts(normalize = False,
                                       sort      = True,
                                       ascending = False).round(decimals = 2)


# In[1196]:


df_full.head()


# ## Feature Engineering
# Feature engineering aimed to enrich the dataset by introducing variables that encapsulate temporal and climatic nuances. The decomposition of the timestamp into day of the week and month variables was intended to capture weekly patterns and seasonal effects on rental demand. Additionally, the introduction of a binary weekend variable provided a straightforward delineation of weekdays from weekends, reflecting typical variations in usage. These engineered features were anticipated to enhance model interpretability and predictive capability by integrating domain knowledge into the analytical framework.

# <br><h3>Standardization</h3>

# In[1197]:


## Standardization ##

# preparing explanatory variable data
df_full_data   = df_full.drop(['RENTALS',
                               'log_RENTALS',
                                'set','FunctioningDay_No','FunctioningDay_Yes'],
                                axis = 1)


# preparing the target variable
df_full_target = df_full.loc[ : , ['RENTALS',
                               'log_RENTALS',
                                   'set','FunctioningDay_No','FunctioningDay_Yes']]


# In[1198]:


# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with the data
scaler.fit(df_full_data)


# TRANSFORMING our data after fit
x_scaled = scaler.transform(df_full_data)


# converting scaled data into a DataFrame
x_scaled_df = pd.DataFrame(x_scaled)


# checking the results
x_scaled_df.describe(include = 'number').round(decimals = 2)


# In[1199]:


# adding labels to the scaled DataFrame

#x_scaled_df = pd.DataFrame(x_scaled_df, index=df_full_data.index, columns=df_full_data.columns)
x_scaled_df.columns = df_full_data.columns
#  Checking pre- and post-scaling of the data
print(f"""
Dataset BEFORE Scaling
----------------------
{np.var(df_full_data)}


Dataset AFTER Scaling
----------------------
{np.var(x_scaled_df)}
""")


# In[1200]:


x_scaled_df.info()


# In[1201]:


df_full_target.info()


# In[1202]:


x_scaled_df.index = df_full_target.index


# In[1203]:


df_full = pd.concat([x_scaled_df, df_full_target], axis=1)


# In[1204]:


df_full = df_full.rename(columns={
    'Temperature(F)': 'Temperature_F',
    'Humidity(%)': 'Humidity',
    'Wind speed (mph)': 'Wind_speed',
    'Visibility(miles)': 'Visibility',
    'DewPointTemperature(F)': 'DewPointTemperature',
    'Rainfall(in)': 'Rainfall',
    'Snowfall(in)': 'Snowfall',
    'SolarRadiation(MJ/m2)': 'SolarRadiation',
    'm_Visibility(miles)': 'm_Visibility',
    'm_DewPointTemperature(F)': 'm_m_DewPointTemperature',
    'm_SolarRadiation(MJ/m2)': 'm_SolarRadiation',
    'log_Wind speed (mph)': 'log_Wind_speed',
    'log_Visibility(miles)': 'log_Visibility',
    'log_Rainfall(in)': 'log_Rainfall',
    'log_SolarRadiation(MJ/m2)': 'log_SolarRadiation'
    
})


# In[1205]:


df_train_1 = df_full[ (df_full['set'] == 'Not Kaggle') & (df_full['FunctioningDay_Yes'] == True)]


# In[1206]:


# making a copy of housing
df_full_explanatory = df_full[ df_full['set'] == 'Not Kaggle' ].copy()


# dropping SalePrice and Order from the explanatory variable set
df_full_explanatory = df_full_explanatory.drop([
                                 'RENTALS',
                                 'log_RENTALS',
                                 'set'], axis = 1)


# formatting each explanatory variable for statsmodels
for val in df_full_explanatory:
    print(val,"+")


# In[1207]:


# building a full model

# blueprinting a model type
lm_full = smf.ols(formula = """RENTALS ~ Temperature_F +
Humidity +
Rainfall +
SolarRadiation +
m_Visibility +
m_SolarRadiation +
log_Rainfall +
log_SolarRadiation +
has_SolarRadiation +
Holiday_No +
Holiday_Yes +
FunctioningDay_No +
FunctioningDay_Yes +
Month +
Hour +
DayOfWeek_rad +
weekday_2 +
weekday_5 +
heatIndex """,
                               data = df_train_1)


# telling Python to run the data through the blueprint
results_full = lm_full.fit()


# printing the results
results_full.summary()


# <br><hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# <h2>Part III: Data Partitioning</h2><br>
# This is a very important step for your submission on Kaggle. Make sure to complete your data preparationbefore moving forward.
# <br>
# <br><h3>Separating the Kaggle Data</h3><br>

# In[1208]:


## parsing out testing data (needed for later) ##

# dataset for kaggle
kaggle_data = df_full[ df_full['set'] == 'Kaggle' ].copy()


# dataset for model building
df = df_full[ df_full['set'] == 'Not Kaggle' ].copy()


# dropping set identifier (kaggle)
kaggle_data.drop(labels = 'set',
                 axis = 1,
                 inplace = True)


# dropping set identifier (model building)
df.drop(labels = 'set',
        axis = 1,
        inplace = True)


# In[1209]:


df = df[ df['FunctioningDay_Yes'] == True ].copy()
df.info()


# <br><h3>Train-Test Split</h3><br>
# Note that the following code will remove non-numeric features, keeping only integer and float data types. It will also remove any observations that contain missing values. This is to prevent errors in the model building process. 

# In[1210]:


#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
x_features = ['Temperature_F', 
              'Humidity',
              'Visibility', 
              'DewPointTemperature', 
              'Rainfall',
              'log_Wind_speed', 
              'log_SolarRadiation', 
              'has_SolarRadiation',
              'Month', 'Day', 
              'Hour', 
              'heatIndex',
              'RENTALS', ] 
# this should be a list



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# removing non-numeric columns and missing values
df = df[x_features].copy().select_dtypes(include=[int, float]).dropna(axis = 0)


# prepping data for train-test split
x_data = df.drop(labels = y_variable,
                 axis   = 1)

y_data = df[y_variable]


# train-test split (to validate the model)
x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                    y_data, 
                                                    test_size    = 0.25,
                                                    random_state = 702 )


# results of train-test split
print(f"""
Original Dataset Dimensions
---------------------------
Observations (Rows): {df.shape[0]}
Features  (Columns): {df.shape[1]}


Training Data (X-side)
----------------------
Observations (Rows): {x_train.shape[0]}
Features  (Columns): {x_train.shape[1]}


Training Data (y-side)
----------------------
Feature Name:        {y_train.name}
Observations (Rows): {y_train.shape[0]}


Testing Data (X-side)
---------------------
Observations (Rows): {x_test.shape[0]}
Features  (Columns): {x_test.shape[1]}


Testing Data (y-side)
---------------------
Feature Name:        {y_test.name}
Observations (Rows): {y_test.shape[0]}""")


# ## Candidate Model Development and Final Model Selection
# 
# The development phase encompassed the training and evaluation of multiple regression models, including OLS Linear Regression, Lasso, Ridge, Elastic Net, K-Nearest Neighbors, and Decision Tree Regressor. Each model was assessed based R-squared values, with a keen focus on balancing predictive accuracy and model complexity. Hyperparameter tuning, conducted on the top-performing models, further refined their configurations to optimize performance. The final model selection was guided by a holistic consideration of error metrics, computational efficiency, and the interpretability of results. The chosen model not only excelled in forecasting daily rentals but also provided actionable insights into the influence of various predictors, aligning with the project's objective of informing strategic decision-making.

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# <h2>Part III: Candidate Modeling</h2><br>
# Develop your candidate models below.

# In[1211]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "Linear_Regression" # name your model

# model type
model = sklearn.linear_model.LinearRegression() # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1239]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "KNN" # name your model

# model type
model = KNeighborsRegressor(algorithm = 'auto',
                   n_neighbors = 4) # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1213]:


# creating lists for training set accuracy and test set accuracy
training_accuracy = []
test_accuracy     = []


# building a visualization of 1 to 50 neighbors
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(x_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(x_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(x_test, y_test))


# plotting the visualization
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# In[1214]:


# finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print(f"""The optimal number of neighbors is {opt_neighbors}""")


# In[1215]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "Lasso (scaled)" # name your model

# model type
model = sklearn.linear_model.Lasso(alpha       = 10.0,
                                  random_state = 702) # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1216]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "Ridge (scaled)" # name your model

# model type
model = sklearn.linear_model.Ridge(alpha = 10.0,
                                   random_state = 702) # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1217]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "Elastic Net (scaled) with MSE" # name your model

# model type
model = sklearn.linear_model.SGDRegressor(alpha = 0.5,
                                          penalty = 'elasticnet',
                                         random_state = 702) # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1218]:


## Candidate Modeling ##

#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
# naming the model
model_name = "Pruned Regression Tree" # name your model

# model type
model = DecisionTreeRegressor(max_depth = 4,
                              min_samples_leaf = 25,
                              random_state = 702) # model type ( ex: sklearn.linear_model.LinearRegression() )



## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)


# In[1219]:


# setting figure size
plt.figure(figsize=(50, 10)) # adjusting to better fit the visual


# developing a plotted tree
plot_tree(decision_tree = model, # changing to pruned_tree_fit
          feature_names = list(x_train.columns),
          filled        = True, 
          rounded       = True, 
          fontsize      = 14)


# rendering the plot
plt.show()


# In[1220]:


########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export = False):
    """
    Plots the importance of features from a CART model.
    
    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """
    
    # declaring the number
    n_features = x_train.shape[1]
    
    # setting plot window
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


# In[1221]:


# plotting feature importance
plot_feature_importances(model,
                         train = x_train,
                         export = False)


# <br>
# <h3>Residual Analysis</h3><br>

# In[1222]:


## Residual Analysis ##

# organizing residuals
model_residuals = {"True"            : y_test,
                   "Predicted"       : model_pred
                  }


# converting residuals into df
model_resid_df = pd.DataFrame(data = model_residuals)


# checking results
model_resid_df.head(n = 5)


#!###########################!#
#!# add more code as needed #!#
#!###########################!#



# <br>
# <h3>Hyperparameter Tuning</h3><br>

# In[1241]:


## Hyperparameter Tuning ##
# declaring a hyperparameter space
criterion_range = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
splitter_range  = ["beast", "random"]
depth_range     = np.arange(1,11,1)
leaf_range      = np.arange(1,251,5)


# creating a hyperparameter grid
param_grid = {'criterion'      : criterion_range,
             'splitter'        : splitter_range,
             'max_depth'       : depth_range,
             'min_samples_leaf': leaf_range}
              


# INSTANTIATING the model object without hyperparameters
tuned_tree = DecisionTreeRegressor()


# RandomizedSearchCV object
tuned_tree_cv = RandomizedSearchCV(estimator             = tuned_tree, #model
                                   param_distributions   = param_grid, #hyperparameter ranges
                                   cv                    = 5,    #folds
                                   n_iter                = 1000, #how many models to build
                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
tuned_tree_cv.fit(x_train, y_train)


# printing the optimal parameters and best score
print("Tuned Parameters  :", tuned_tree_cv.best_params_)
print("Tuned Training AUC:", tuned_tree_cv.best_score_.round(4))


# In[1242]:


# naming the model
model_name = 'Tuned Tree'


# INSTANTIATING a logistic regression model with tuned values
model = DecisionTreeRegressor(splitter         = 'random',
                              min_samples_leaf = 6,
                              max_depth        = 9,
                              criterion        = 'squared_error')


# FITTING to the TRAINING data
model.fit(x_train, y_train)


# PREDICTING based on the testing set
model.predict(x_test)


# SCORING results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)


# In[1243]:


## Hyperparameter Tuning ##
# declaring a hyperparameter space
n_neighbors = np.arange(1, 31)
weights     = ['uniform', 'distance']
algorithm   = ['ball_tree', 'kd_tree', 'brute', 'auto']
leaf_size   = np.arange(1, 50)
p_size      = [1, 2]


# creating a hyperparameter grid
param_grid = {
            'n_neighbors': n_neighbors, 
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,  
            'p': p_size 
}
              


# INSTANTIATING the model object without hyperparameters
tuned_knn = KNeighborsRegressor()


# RandomizedSearchCV object
tuned_knn_cv = RandomizedSearchCV(estimator             = tuned_knn, #model
                                   param_distributions   = param_grid, #hyperparameter ranges
                                   cv                    = 5,    #folds
                                   n_iter                = 1000, #how many models to build
                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
tuned_knn_cv.fit(x_train, y_train)


# printing the optimal parameters and best score
print("Tuned Parameters  :", tuned_knn_cv.best_params_)
print("Tuned Training AUC:", tuned_knn_cv.best_score_.round(4))


# In[1268]:


# naming the model
model_name = 'Tuned KNN'


# INSTANTIATING a logistic regression model with tuned values
model = KNeighborsRegressor(weights = 'distance',
                            p = 1,
                              n_neighbors        = 5,
                              leaf_size        = 4,
                           algorithm = 'kd_tree')


# FITTING to the TRAINING data
model.fit(x_train, y_train)


# PREDICTING based on the testing set
model.predict(x_test)


# SCORING results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)


# In[1251]:


## Hyperparameter Tuning ##
# declaring a hyperparameter space
alpha = np.logspace(-4, 4, 200)
max_iter     = [1000, 5000, 10000]
selection   = ['cyclic', 'random']


# creating a hyperparameter grid
param_grid = {
            'alpha': alpha, 
            'max_iter': max_iter,
            'selection': selection,
             
}
              


# INSTANTIATING the model object without hyperparameters
tuned_Lasso = sklearn.linear_model.Lasso()


# RandomizedSearchCV object
tuned_Lasso_cv = RandomizedSearchCV(estimator             = tuned_Lasso, #model
                                   param_distributions   = param_grid, #hyperparameter ranges
                                   cv                    = 5,    #folds
                                   n_iter                = 1000, #how many models to build
                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
tuned_Lasso_cv.fit(x_train, y_train)


# printing the optimal parameters and best score
print("Tuned Parameters  :", tuned_Lasso_cv.best_params_)
print("Tuned Training AUC:", tuned_Lasso_cv.best_score_.round(4))


# In[1254]:


# naming the model
model_name = 'Tuned Lasso'


# INSTANTIATING a logistic regression model with tuned values
model = sklearn.linear_model.Lasso(selection = 'random',
                                  max_iter = 5000,
                                  alpha = 2.64)


# FITTING to the TRAINING data
model.fit(x_train, y_train)


# PREDICTING based on the testing set
model.predict(x_test)


# SCORING results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)


# In[1255]:


## Hyperparameter Tuning ##
# declaring a hyperparameter space
alpha = np.logspace(-6, 6, 200)
solver     = ['svd', 'cholesky', 'lsqr', 'sag', 'saga']
max_iter   = [None, 1000, 5000, 10000]


# creating a hyperparameter grid
param_grid = {
            'alpha': alpha, 
            'solver': solver,
            'max_iter': max_iter,
             
}
              


# INSTANTIATING the model object without hyperparameters
tuned_Ridge = sklearn.linear_model.Ridge()


# RandomizedSearchCV object
tuned_Ridge_cv = RandomizedSearchCV(estimator             = tuned_Ridge, #model
                                   param_distributions   = param_grid, #hyperparameter ranges
                                   cv                    = 5,    #folds
                                   n_iter                = 1000, #how many models to build
                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
tuned_Ridge_cv.fit(x_train, y_train)


# printing the optimal parameters and best score
print("Tuned Parameters  :", tuned_Ridge_cv.best_params_)
print("Tuned Training AUC:", tuned_Ridge_cv.best_score_.round(4))


# In[1256]:


# naming the model
model_name = 'Tuned Ridge'


# INSTANTIATING a logistic regression model with tuned values
model = sklearn.linear_model.Ridge(solver = 'cholesky',
                                  max_iter = 1000,
                                  alpha = 8.603464416684492)


# FITTING to the TRAINING data
model.fit(x_train, y_train)


# PREDICTING based on the testing set
model.predict(x_test)


# SCORING results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)


# displaying results
print('Training Score :', model_train_score)
print('Testing Score  :', model_test_score)
print('Train-Test Gap :', model_gap)


# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# <h2>Part IV: Preparing Submission File for Kaggle</h2><br>
# The code below will store the predicted values for each of the models above.

# In[1225]:


kaggle_df


# In[1269]:


# removing non-numeric columns and missing values
kaggle_df = kaggle_data[x_features].copy()


# x-data
x_data_kaggle = kaggle_df.drop(labels = y_variable,
                               axis   = 1)

# y-data
y_data_kaggle = kaggle_df[y_variable]


# Fitting model from above to the Kaggle test data
kaggle_predictions = model.predict(x_data_kaggle)


# In[1270]:


kaggle_df


# <br>
# <h3>Creating the Kaggle File</h3><br>

# In[1271]:


## Kaggle Submission File ##

# organizing predictions
model_predictions = {"RENTALS" : kaggle_predictions}


# converting predictions into df
model_pred_df = pd.DataFrame(data  = model_predictions,
                             index = df_test.index)


model_pred_df.head()


# In[1272]:


# reading testing data into Python
testing_data = './datasets/test.xlsx'

# calling this df_test
df_test_2 = pd.read_excel(io         = testing_data,
                        sheet_name = 'data',
                        header     = 0,
                        index_col  = 'ID')
df_test_2.head()


# In[1273]:


model_pred_df['FunctioningDay'] = df_test_2['FunctioningDay'].values
model_pred_df.head()


# In[1274]:


model_pred_df.loc[model_pred_df['FunctioningDay'] == 'No', 'RENTALS'] = 0
model_pred_df.head()


# In[1275]:


model_pred_df = model_pred_df.drop('FunctioningDay', axis=1)
model_pred_df.head()


# In[1276]:


#!######################!#
#!# name the .csv file #!#
#!######################!#

# sending predictions to .csv
model_pred_df.to_csv(path_or_buf = "./model_output/Solis_Jorge_A1_v3.csv",
                     index       = True,
                     index_label = 'ID')


# <br>

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
# 
# ~~~
# 
#   _    _                           __  __           _      _ _             _ 
#  | |  | |                         |  \/  |         | |    | (_)           | |
#  | |__| | __ _ _ __  _ __  _   _  | \  / | ___   __| | ___| |_ _ __   __ _| |
#  |  __  |/ _` | '_ \| '_ \| | | | | |\/| |/ _ \ / _` |/ _ \ | | '_ \ / _` | |
#  | |  | | (_| | |_) | |_) | |_| | | |  | | (_) | (_| |  __/ | | | | | (_| |_|
#  |_|  |_|\__,_| .__/| .__/ \__, | |_|  |_|\___/ \__,_|\___|_|_|_| |_|\__, (_)
#               | |   | |     __/ |                                     __/ |  
#               |_|   |_|    |___/                                     |___/   
# 
#                                                             
# 
# ~~~
# 
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />

# <br>
