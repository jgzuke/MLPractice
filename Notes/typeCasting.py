import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
# rng.rand(a, b, c) = a x b x c array of randoms

X = rng.rand(10, 2000)
# X dataytpe (X.dtype) = 'float64'

X = np.array(X, dtype='float32')
# cast array to dtype = 'float32'

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
# type back to 'float64'


from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)  
list(clf.predict(iris.data[:3]))
# prints list of indices

# target_names is names by index, target_names[target] gives list of names
clf.fit(iris.data, iris.target_names[iris.target]) 
list(clf.predict(iris.data[:3])) 
# prints list of names because trained on names