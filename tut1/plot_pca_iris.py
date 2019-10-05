import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go

centers = [[1, 1], [-1, -1], [1, -1]]
df = pd.read_csv('./iris.csv', names=['sepal length','sepal width','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
X = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
X = StandardScaler().fit_transform(X)

# components=3
fig = plt.figure(figsize = (8,8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
pca = decomposition.PCA(n_components=3)
PA3 = pca.fit_transform(X)
principalDf = pd.DataFrame(data = PA3, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
targets = ['Setosa', 'Versicolor', 'Virginica']
colors = ['r', 'g', 'b']
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
   , finalDf.loc[indicesToKeep, 'principal component 2']
   , finalDf.loc[indicesToKeep, 'principal component 3']
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


#componets=2
pca = decomposition.PCA(n_components=2)
PA2 = pca.fit_transform(X)
principalDf = pd.DataFrame(data = PA2, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
for target, color in zip(targets,colors):
  indicesToKeep = finalDf['target'] == target
  ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
   , finalDf.loc[indicesToKeep, 'principal component 2']
   , c = color
   , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
