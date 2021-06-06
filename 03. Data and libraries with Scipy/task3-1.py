"""
@author: Kovarthanan K
"""
#importing needed library 
import pandas as pd
import numpy as np

#importing .csv file 
df = pd.read_csv('result.csv') #stores as dataframe 
results = df.to_numpy() #converting to numpy array 

def column_data(col_name, col_index):
    print("{} average : {}".format(col_name, np.mean(results[:,col_index])))
    print("{} min : {}".format(col_name, np.min(results[:,col_index])))
    print("{} max : {}".format(col_name, np.max(results[:,col_index])))
    print()
    
column_data("ass1", 1)
column_data("ass2", 2)
column_data("ass3", 3)
column_data("ass4", 4)
column_data("exam", 5)
column_data("total", 6)

#printing student info with the highest total 
print(df.loc[df['Total'] == 91.0].to_string(index=False))