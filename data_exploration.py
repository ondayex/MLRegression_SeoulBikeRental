# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# load dataset，check for missing values and see the data types
dataset = pd.read_csv('dataset/SeoulBikeData.csv')
dataset.info()
print(dataset.head(30))

# %%
#further check on categorical features
print(dataset['Seasons'].unique())
print(dataset['Holiday'].unique())
print(dataset['Functioning Day'].unique())

# %%
#Statistical summary of numerical features (central tendencies, spread, and shape of the data)
print(dataset.iloc[:, [2,3,4,5,6,7,8,9,10]].describe())

dataset[['Hour','Temperature','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']].hist(bins=20, edgecolor='black', figsize=(12, 6))
plt.show()
# %%
#Statistical summary of categorical features (distribution)
print(dataset.iloc[:, [11,12,13]].describe())
# Plot bar plot
selected_columns = dataset.iloc[:, [11, 12, 13]]
column_names = selected_columns.columns

# Create a plot for each column
for column in column_names:
    # Count the occurrences of each category
    category_counts = selected_columns[column].value_counts()
    
    # Plot the bar plot
    plt.figure()  # Create a new figure for each plot
    category_counts.plot(kind='bar', edgecolor='black', color='skyblue')
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()
# %%
#process categorical features
Seasons_dummies = pd.get_dummies(dataset['Seasons'], drop_first = True, dtype = int)
dataset = pd.concat([dataset, Seasons_dummies], axis = 1)
dataset.drop(['Seasons'], axis = 1, inplace = True)

dataset['Holiday'] = dataset['Holiday'].apply(lambda x: 1 if x == 'Holiday' else 0)

dataset['Functioning Day'] = dataset['Functioning Day'].apply(lambda x: 1 if x == 'Yes' else 0)

print(dataset.head(30))

# %%
#split data into train and test
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X, y)

