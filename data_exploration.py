# %%
import pandas as pd

# %%
# load dataset and check for missing values and see the data types
dataset = pd.read_csv('dataset/SeoulBikeData.csv')
dataset.info()
print(dataset.head(30))

# %%
#further check on categorical features
print(dataset['Seasons'].unique())
print(dataset['Holiday'].unique())
print(dataset['Functioning Day'].unique())

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

