"""
@author: Kovarthanan K
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#importing .csv file
df = pd.read_csv('Malicious_or_criminal_attacks_breakdown-Top_five_industry_sectors_July-Dec-2019.csv - Sheet1.csv', index_col=0, engine='python')

#changing rows into column and column into row
altered_df =df.transpose()

#colours for bars
colors = ['red', 'yellow', 'blue', 'green']

#defining subplots 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), dpi=100)

#Grouped bar Chart 
grouped_bar_chart = altered_df.plot.bar(ax=axes[0], rot=90, color=colors, 
                                        ylabel='Number of attack', 
                                        xlabel='the top five industry sectors', 
                                        title='Type of attack by top five industry sectors')

#Stacked Bar Chart
stacked_bar_chart = altered_df.plot.bar(ax=axes[1], rot=90, color=colors, 
                                        ylabel='Number of attack', 
                                        xlabel='the top five industry sectors', 
                                        title='Type of attack by top five industry sectors', stacked=True)
