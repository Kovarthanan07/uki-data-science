"""
@author: Kovarthanan K
"""
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import PCA

# data
df = pd.read_csv('spambase.data')

corr = df.corr()

df_feature = df.loc[:, :'278']
df_label = df['1']
train_x, test_x, train_y, test_y = train_test_split(df_feature, df_label)

# desicion tree with basic hyperparameters
dt = DecisionTreeClassifier()
# fit the train data for model
dt.fit(train_x, train_y)
# predict the test data with model
y_pred = dt.predict(test_x)
# accuracy of the model
print ("DecisionTreeClassifier accuracy : {}".format(accuracy_score(test_y, y_pred)))

# random forest classifier with basic hyperparameters
rf = RandomForestClassifier()
# fit the train data for model
rf.fit(train_x, train_y)
# predict the test data with model
y_pred1 = rf.predict(test_x)
# accuracy of the model
print("RandomForestClassifier Accuracy : {}".format(accuracy_score(test_y, y_pred1)))

# support vector machine with basic hyperparameters
svm = SVC()
# fit the train data for model
svm.fit(train_x, train_y)
# predict the test data with model
y_pred2 = svm.predict(test_x)
# accuracy of the model
print("SVM Accuracy : {}".format(accuracy_score(test_y, y_pred2)))

# features are in different distribution so I normalize the feature distribution 
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_feature)
df_normalized = pd.DataFrame(scaled_values)

train_x1, test_x1, train_y1, test_y1 = train_test_split(df_normalized, df_label)

# desicion tree with basic hyperparameters
dt = DecisionTreeClassifier()
# fit the train data for model
dt.fit(train_x1, train_y1)
# predict the test data with model
y_pred = dt.predict(test_x1)
# accuracy of the model
print ("DecisionTreeClassifier accuracy with basic hyperparameters : {}".format(accuracy_score(test_y, y_pred)))

# Random forest classifier with basic hyperparameters
rf = RandomForestClassifier()
# train the model with normalized train data
rf.fit(train_x1, train_y1)
# predict the model with normalized test data 
y_pred1 = rf.predict(test_x1)
# accuracy of the model
print("RandomForestClassifier Accuracy  with basic hyperparameters : {}".format(accuracy_score(test_y, y_pred1)))

# support vector machine with basic hyperparameters
svm1 = SVC()
# train the normalized data 
svm1.fit(train_x1, train_y1)
# predict the model with normalized test data
y_pred = svm1.predict(test_x1)
# accuracy of the model
print("SVM Accuracy with basic hyperparameters : {}".format(accuracy_score(test_y, y_pred2)))

# support vector machine with following hyperparameters C=11 and gamma=5 (manually not use the grid search)
svm1 = SVC(C=11, gamma=5)
# train the model with normalized train data
svm1.fit(train_x1, train_y1)
# predict the model with normalized test data
y_pred = svm1.predict(test_x1)
# accuracy of the model
accuracy_score(test_y1, y_pred)

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

# set the grid search parameters
hyper_params = {'n_estimators': n_estimators, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}

# create basic random forest classifier model
rf = RandomForestClassifier(random_state=1)
# create grid search for find the best hyper parameters
grid_rf = GridSearchCV(estimator=rf, param_grid=hyper_params, cv=3, n_jobs=-1, verbose=1)
# fit the normalized train data for best model
grid_rf.fit(train_x1, train_y1)
# predict the model with normalized test data
y_pred_1 = grid_rf.predict(test_x1)
# accuracy of the model
accuracy_score(test_y1, y_pred1)

# feature importances of above best ramdom forest classifier model
feature_importances = grid_rf.best_estimator_.feature_importances_

# plot the feature importances
fig, ax = plt.subplots(figsize=(7,7), dpi=100)
ax.barh(df_feature.columns, feature_importances)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')