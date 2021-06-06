"""
@author: Kovarthanan K
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set()

# import data
df = pd.read_csv('task6_1_dataset.csv')

# KNN model with 1 neighbors
knn = KNeighborsClassifier(n_neighbors=1)
# fitting data for KNN model
knn.fit(np.c_[df.loc[:, 'x1':'x2']], df['y'])

colors = ['green', 'blue', 'magenta']
cmap = ListedColormap(colors)

pred = knn.predict([(-4, 8)])

# plot the scatters for each group seperate color
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.scatter(df['x1'], df['x2'], c=df['y'], cmap=cmap, s=75, alpha=0.6)
# plot the mark for predict data
ax.scatter(-4, 8, marker='x', s=200, color=colors[int(pred[0])], lw=2)
# give annotation for predict mark
ax.annotate('(-4,8), Test point', (-4,8), color='red')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('3-Class classification (k=1) \n the test point is predicted as class green')
fig.savefig('task6-1-KNN')

# KNN model with 15 neighbors
knn2 = KNeighborsClassifier(n_neighbors=15)
# fit the data for KNN model
knn2.fit(np.c_[df.loc[:, 'x1':'x2']], df['y'])

# predict the data using knn2 model
y_pred2 = knn2.predict([[-2, 5]])

cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA'])
colors = ['green', 'blue', 'magenta']
cmap_bold = ListedColormap(colors)

# create axis the decision boundary
h = 0.05
x1_min, x1_max = df['x1'].min() - 1, df['x1'].max() + 1
x2_min, x2_max = df['x2'].min() - 1, df['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
y_pred3 = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
z = y_pred3.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
# plot decision boundary
ax.pcolormesh(xx, yy, z, cmap=cmap_light)
# plot each group in seperate color
ax.scatter(df['x1'], df['x2'], c=df['y'], cmap=cmap_bold, alpha=0.6, s=75)
ax.scatter(-2, 5, marker='x', lw=2, c=colors[int(y_pred2[0])], s=150)
ax.annotate('(-2, 5) test point', [-2,5], c='red')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('3-Class classification (k=15) \n the test point is predicted as class magenta')
fig.savefig('task6-1-decision_boundary')

