from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
prediction = clf.predict(X[0:1])
print(str(prediction) + " (should be " + str(y[0:1]) + ")")

import pickle
learnedModel = pickle.dumps(clf)
clf2 = pickle.loads(learnedModel)
prediction = clf2.predict(X[0:1])
print(str(prediction) + " (should be " + str(y[0:1]) + ")")

from sklearn.externals import joblib
joblib.dump(clf2, 'PickleModels/filename.pkl')
clf3 = joblib.load('PickleModels/filename.pkl')
prediction = clf3.predict(X[0:1])
print(str(prediction) + " (should be " + str(y[0:1]) + ")")