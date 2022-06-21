
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from numpy.random import randn
#import xgboost

# COMMENT CTRL+SHIFT+/

# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# print(iris)
# print(X)
# print(y)


header_list = ["sepal length", "sepal width", "petal length",
               "petal width", "type"]


iris = pd.read_csv("/home/mpa/Downloads/iris.csv", 
                   names = header_list)

iris.info()
iris.head()

# Set default Seaborn style
sns.set()

_ = sns.pairplot(iris, hue = 'type')

iris.type.value_counts()
iris['sepal length'].describe(percentiles = [0.5,0.97])
iris.describe(exclude = [np.number])

iris.describe(exclude = [object], percentiles = [0.01, 0.1, 0.3, 0.97]) 

grupo_tipo_mediana = iris.groupby('type').median()
grupo_tipo_mediana


type(grupo_tipo_mediana)

grupo = iris.groupby('type')


type(grupo)
list(grupo)

list(grupo)[2]


'''
dessa forma, filtra-se a coluna TYPE somente com a categoria "Iris_virginica" e 
simultaneamente apenas para os valores da coluna PETAL LENGTH
'''
virginica_petal_length = iris[iris['type'] == 'Iris-virginica']['petal length']
virginica_petal_length = virginica_petal_length.to_numpy()


type(virginica_petal_length)

# Compute number of data points: n_data
n_data = len(virginica_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
plt.hist(virginica_petal_length, bins = n_bins)

# Label axes
_ = plt.xlabel('petal length')
_ = plt.ylabel('count')

# Show histogram
plt.show()

'''
dessa forma, filtra-se a coluna TYPE somente com a categoria "Iris-setosa" e 
simultaneamente apenas para os valores da coluna PETAL LENGTH
'''
setosa_petal_length = iris[iris['type'] == 'Iris-setosa']['petal length']
setosa_petal_length = setosa_petal_length.to_numpy()

# Plot histogram of setosa petal lengths
_ = plt.hist(setosa_petal_length)

'''
dessa forma, filtra-se a coluna TYPE somente com a categoria "Iris-versicolor" e 
simultaneamente apenas para os valores da coluna PETAL LENGTH
'''
versicolor_petal_length = iris[iris['type'] == 'Iris-versicolor']['petal length']
versicolor_petal_length = versicolor_petal_length.to_numpy()

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

# Compute ECDF for virginica data: x_vers, y_vers
x_vers, y_vers = ecdf(virginica_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


# Compute ECDF for setosa data: x_vers, y_vers
x_vers, y_vers = ecdf(setosa_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
_ = plt.legend(('setosa', 'versicolor', 'virginica'), loc = 'lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

iris[np.logical_and(iris["petal length"] > 4, iris["petal length"] < 6)]

_ = iris["type"].value_counts().plot(kind = "barh")


_ = iris["type"].value_counts().plot(kind = "bar")



























































