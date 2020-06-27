import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import seaborn as sns
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "insurance.csv")

#import data
data = pd.read_csv(path)

#review the data
print(data.head())


## Handling Missing Values

#check how many values are missing (NaN) before we apply the methods below 
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#fill in the missing values

#option 0 for dropping the entire column
data = pd.read_csv(path)
data.drop('bmi', axis = 1, inplace = True)
#check how many values are missing (NaN) - after we dropped 'bmi'
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option 1 for dropping NAN
data = pd.read_csv(path)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option 2 for filling NaN
data = pd.read_csv(path)
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])

#option 3 for filling NaN
data = pd.read_csv(path)
data['bmi'].fillna(data['bmi'].mean(), inplace = True)
print(data.head())
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])


## After filling in, now let's visualize

figure, ax = plt.subplots(3,2, figsize=(12,24))

#See the distributions of the data
sns.distplot(data['charges'],ax= ax[0,0])
sns.distplot(data['age'],ax=ax[0,1])
sns.distplot(data['bmi'],ax= ax[1,0])
sns.distplot(data['children'],ax= ax[1,1])

sns.countplot(data['smoker'],ax=ax[2,0])
sns.countplot(data['region'],ax= ax[2,1])


#Visualizeing skewness
sns.pairplot(data)

#Smokers vs non-smokers on age vs charges

sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'muted', height = 7)
plt.show(sns)

#Correlation between these factors

corr = data.corr()

sns.heatmap(corr, cmap = 'Wistia', annot= True)
plt.show(sns)
