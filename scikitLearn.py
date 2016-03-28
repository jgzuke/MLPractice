from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
# digits.data = X or array of features
# digits.target = Y or array of wanted results
# digits.images[i] is 8x8 version of data[i]

# >>> digits.target.size
# 1797
# >>> digits.data.size
# 115008
# >>> digits.images.size
# 115008
# >>> digits.images[0].size
# 64
# >>> digits.data[0].size
# 64

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

predictions = clf.predict(digits.data[-1:])
print(str(predictions) + " (should be " + str(digits.target[-1:]) + ")")
# Predicts 8