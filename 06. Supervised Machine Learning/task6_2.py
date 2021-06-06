"""
@author: Kovarthanan K
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('payment_fraud.csv')

# one-hot encoding
df = pd.get_dummies(df, columns=['paymentMethod'])

# split data as train and test data (features, labels)
columns = ['accountAgeDays', 'numItems', 'localTime', 'paymentMethodAgeDays', 'paymentMethod_creditcard', 'paymentMethod_paypal', 'paymentMethod_storecredit']
y = df['label']
x = df[columns]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33)

# create C=1 logistic regression model
lor = LogisticRegression()
lor.fit(train_x, train_y)

# predict C=1 logistic regression model
lor.predict(test_x)

# test set and train set score of C=1 logistic regression
lor_train_score = lor.score(train_x, train_y)
lor_test_score = lor.score(test_x, test_y)
print('train set score of the c=1 logistic regression model', lor_train_score)
print('test set score of the c=1 logistic regression model', lor_test_score)

# create C=100 logistic regression model
lor_100 = LogisticRegression(C=100)
lor_100.fit(train_x, train_y)

# test set and train set score of C=100 logistic regression
lor100_train_score = lor_100.score(train_x, train_y)
lor100_test_score = lor_100.score(test_x, test_y)
print('train set score of the c=100 logistic regression model', lor100_train_score)
print('test set score of the c=100 logistic regression model', lor100_test_score)

# create C=10 logistic regression model
lor_10 = LogisticRegression(C=10)
lor_10.fit(train_x, train_y)

# test set and train set score of C=10 logistic regression
lor10_train_score = lor_10.score(train_x, train_y)
lor10_test_score = lor_10.score(test_x, test_y)
print('train set score of the c=10 logistic regression model', lor10_train_score)
print('test set score of the c=10 logistic regression model', lor10_test_score)

# create C=0.001 logistic regression model
lor_001 = LogisticRegression(C=0.01)
lor_001.fit(train_x, train_y)

# test set and train set score of C=0.001 logistic regression
lor001_train_score = lor_001.score(train_x, train_y)
lor001_test_score = lor_001.score(test_x, test_y)
print('train set score of the c=0.001 logistic regression model', lor001_train_score)
print('test set score of the c=0.001 logistic regression model', lor001_test_score)

# create C=0.01 logistic regression model
lor_01 = LogisticRegression(C=0.1)
lor_01.fit(train_x, train_y)

# test set and train set score of C=0.01 logistic regression
lor01_train_score = lor_01.score(train_x, train_y)
lor01_test_score = lor_01.score(test_x, test_y)
print('train set score of the c=0.1 logistic regression model', lor01_train_score)
print('test set score of the c=0.1 logistic regression model', lor01_test_score)

# plot all logistic regression model which different C coefficient 
fig, ax = plt.subplots(figsize=(7,7), dpi=100)
ax.scatter(columns, lor.coef_.T, label='C=1', alpha=0.8)
ax.scatter(columns, lor_10.coef_.T, marker='*', label='C=10', alpha=0.8)
ax.scatter(columns, lor_100.coef_.T, marker='v', label='C=100', alpha=0.8)
ax.scatter(columns, lor_01.coef_.T, marker='^', label='C=0.01', alpha=0.8)
ax.scatter(columns, lor_001.coef_.T, marker='p', label='C=0.001', alpha=0.8)
ax.set_ylim([5,-5])
ax.axhline(y=0, c='k')
ax.set_xlabel("Feature")
ax.set_ylabel("Coefficient magnitude")
ax.legend()
ax.set_xticklabels(columns, rotation=90, ha='center') 
fig.savefig('6_2_logist_coefficient')

# decision tree model
clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)

# test set and train set score of decision tree model clf
clf_train_score = clf.score(train_x, train_y)
clf_test_score = clf.score(test_x, test_y)
print('train set score of the Decision Tree Classifier', clf_train_score)
print('test set score of the Decision Tree Classifier', clf_test_score)

# feature importances
feature_importances = clf.feature_importances_
print('Decision tree feature importances', feature_importances)
print('decision tree depth', clf.max_depth)

# plot horizontal bar chart to plot feature importances of the decision tree
fig, ax = plt.subplots(figsize=(7,7), dpi=100)
ax.barh(columns, feature_importances)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
fig.savefig('6_2_feature_importances')


