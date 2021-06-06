"""
@author: Kovarthanan K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('admission_predict.csv')


def linearRegressionDetails(x_data, y_data, color):
    x = df[x_data].values
    y = df[y_data].values
    
    #splitting train & test data
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25, shuffle = False)
    length_train = 300
    x_train = x_train.reshape(length_train, 1)
    y_train = y_train.reshape(length_train, 1)
    length_test = 100
    x_test = x_test.reshape(length_test, 1)
    y_test = y_test.reshape(length_test, 1)
    
    #creating model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    r_sq = model.score(x_test, y_test)
    print('coefficient of determination:', r_sq)
    
    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b)
    plt.ylabel("Y (" + y_data + ")")
    plt.xlabel("X ("+x_data+")")
    plt.title("Linear regression with "+x_data+" "+y_data)
    plt.show()




linearRegressionDetails('GRE Score', 'Chance of Admit', 'red')
linearRegressionDetails('CGPA', 'Chance of Admit', 'green')