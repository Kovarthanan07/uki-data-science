"""
@author: Kovarthanan K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'a': np.random.randint(0, 50, size=100)})
df['b'] = df['a'] + np.random.normal(0, 10, size=100)
df['c'] = 100 - 2* df['a'] + np.random.normal(0, 10, size=100)
df['d'] = np.random.randint(0, 50, 100)

def scatterPlot_correlation(data_x, data_y, color):
    x=df[data_x]
    y=df[data_y]
    line_coef = np.polyfit(x,y,1)
    xx = np.arange(0, 50, 0.1)
    yy = line_coef[0]*xx + line_coef[1]
    plt.scatter(x, y)
    plt.xlabel(data_x)
    plt.ylabel(data_y)
    plt.plot(xx, yy,color, lw=2)
    plt.show()
    correlation = np.corrcoef(x, y)
    print ("{} and {} pearson_r : {} ".format(data_x, data_y, correlation[0][1]))
    print ("{} and {} corrcoef : {} ".format(data_x, data_y, correlation))
    
scatterPlot_correlation('a', 'b', 'red')
scatterPlot_correlation('a', 'c', 'green')
scatterPlot_correlation('a', 'd', 'blue')
