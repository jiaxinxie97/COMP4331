import pandas as pd
#import dataset
iris = pd.read_csv('iris.csv',names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())
import matplotlib.pyplot as plt

#scatter plot
colors = {'Setosa':'r', 'Versicolor':'g', 'Virginica':'b'}
# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
for i in range(len(iris['sepal_length'])):
    ax.scatter(iris['sepal_length'][i], iris['sepal_width'][i],color=colors[iris['class'][i]])
# set a title and labels
ax.set_title('Scatter Plot')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
plt.show()

#Line Chart
columns = iris.columns.drop(['class'])
# create x data
x_data = range(0, iris.shape[0])
# create figure and axis
fig, ax = plt.subplots()
# plot each column
for column in columns:
    ax.plot(x_data, iris[column])
# set title and legend
ax.set_title('Line Chart')
plt.xlabel('index')
plt.ylabel('length unit')
ax.legend(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()

#Histogram
plt.figure()
x = iris["sepal_length"]
plt.subplot(221)
plt.hist(x, bins = 20, color = "green")
plt.title("Histogram for sepal_length")
plt.xlabel("sepal_length")
plt.ylabel("Count") 

x = iris["sepal_width"]
plt.subplot(222)
plt.hist(x, bins = 20, color = "orange")
plt.title("Histogram for sepal_width")
plt.xlabel("sepal_width")
plt.ylabel("Count")

x = iris['petal_length']
plt.subplot(223)
plt.hist(x, bins = 20, color = "red")
plt.title("Histogram for petal_length")
plt.xlabel("petal_length")
plt.ylabel("Count")

x = iris['petal_width']
plt.subplot(224)
plt.hist(x, bins = 20, color = "blue")
plt.title("Histogram for petal_width")
plt.xlabel("petal_width")
plt.ylabel("Count")
plt.show()

#different bins histogram
plt.figure()
x = iris["sepal_length"]
plt.subplot(121)
plt.hist(x, bins = 10, color = "green")
plt.title("10 bins Histogram")
plt.xlabel("sepal_length")
plt.ylabel("Count")

x = iris["sepal_length"]
plt.subplot(122)
plt.hist(x, bins = 20, color = "green")
plt.title("20 bins Histogram")
plt.xlabel("sepal_length")
plt.ylabel("Count")
plt.show()

#Box plot
plt.figure()
new_iris=iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]] 
new_iris.boxplot()
plt.title('Box Plot')
plt.show()


