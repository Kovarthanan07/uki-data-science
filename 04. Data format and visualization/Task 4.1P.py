"""
@author: Kovarthanan K
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt

#setting datas for bar chart
x = ['Cyber incident', 'Theft of paperwork or data storage device', 'Rogue employee', 'Social engineering / impersonation']
y = [108,32,26,11]

#setting up Bar chart 
plt.bar(x, y, align='center', alpha=0.5, color=['blue', 'red', 'green', 'yellow'])
plt.xticks(x, x, rotation ='vertical')
plt.ylabel('number of attacks per attack type')
plt.xlabel('attack type')
plt.title('Number of malicious or criminal attack July-December-2019')
plt.figure(figsize=(7,5), dpi=100)
plt.show()

#setting datas for pie chart 
data_pie = np.array([63, 40, 30, 30, 14])
label_pie = ['Health service providers', 'Finance', 'Education', 'Legal accounting & management services', 'Personal services']

#functon for pie chart
def pie_chart(data, label): 
    plt.pie(data, labels = label, autopct='%1.0f%%')
    plt.figure(figsize=(10, 10))
    plt.show() 
    
pie_chart(data_pie, label_pie)
