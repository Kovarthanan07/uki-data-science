"""
@author: Kovarthanan K
"""
import pandas as pd

#importing .csv file 
df = pd.read_csv('result_withoutTotal.csv') #stores as dataframe 
results = df.to_numpy() #converting to numpy array 

#calculating total 
df['Total'] = (df['Ass1'] + df['Ass3'])*0.05 + (df['Ass2'] + df['Ass4'])*0.15 +df['Exam']*0.6  
df.loc[df['Total'] > 100.0, 'Total'] = 100.0

#Setting up Final column 
df['Final'] = df['Total']
df.Final = df.Final.astype(float).round().astype(int)
df.loc[((df['Exam'] < 40) & (df['Final'] >= 44)), 'Final'] = 44

#setting up Grade Column
grade_points = [0, 49.45, 59.45, 69.45, 79.45, 100]
grade_letters = ['N', 'P', 'C', 'D', 'HD']
grade_calc = pd.cut(df.Final, grade_points, labels=grade_letters)
df['Grade'] = grade_calc

#exporing result_updated.txt file
df.to_csv('result_updated.txt')

#exporing failedhurdle.txt file
failed_details = df.loc[df.Exam <48]
failed_details.to_csv('failedhurdle.txt')

#displaying needed output
print(df)
print("students with exam score < 48")
print(failed_details.to_string(index=False))
print('students with exam score > 100')
print(df.loc[df['Final'] == 100].to_string(index=False))