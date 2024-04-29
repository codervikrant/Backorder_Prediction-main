# Import libraries 
import pandas as pd 
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function

# Import dataset 
df = pd.read_csv('data/cleaned_data.csv')

le = LabelEncoder()
df['product'] = le.fit_transform(df['product'])
#print(df['product'].unique())

# Converting Pandas DataFrame into a Numpy array
X = df.drop(['went_on_backorder', 'company'], axis= 1).values
y = df['went_on_backorder'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18) # 70% training and 30% test

param_dist = {'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100], 
              'max_depth': [None, 2, 5, 7, 9, 10, 12, 15, 17, 20, 30, 50]}

#Create a Random Forest Classifier
clf = RandomForestClassifier(random_state=18)

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(clf, param_distributions = param_dist, scoring = 'roc_auc', n_iter=5, n_jobs = -1, cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_n_estimator = rand_search.best_estimator_.n_estimators
best_max_depth = rand_search.best_estimator_.max_depth

#Create a Random Forest Classifier
clf = RandomForestClassifier(criterion = 'gini', n_estimators = best_n_estimator , max_depth = best_max_depth)
clf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(clf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
