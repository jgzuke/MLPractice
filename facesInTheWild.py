# Labaled faces in the wild
# http://nbviewer.jupyter.org/github/ogrisel/notebooks/blob/master/Labeled%20Faces%20in%20the%20Wild%20recognition.ipynb
# terminal dump

# Plotting
# Loading
# Shaping

# Plots pictures with titles
def plot_gallery(images, titles, h, w, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.7 * n_col, 2.3 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

# Creates titles for pictures by what was true and guessed
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

import pylab as pl
import numpy as np

from sklearn.datasets import fetch_lfw_people

# Grab dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.data
y = lfw_people.target
names = lfw_people.target_names

examples, h, w = lfw_people.images.shape
features = h * w
classes = len(names)

# plot peoples pictures
plot_gallery(X, names[y], h, w)

# Create a new empty graph
pl.figure(figsize=(14, 3))

# find unique elements http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.unique.html
y_unique = np.unique(y)
# count y's equal to i
counts = [(y == i).sum() for i in y_unique]

# plot x labels
pl.xticks(y_unique,  names[y_unique])
locs, labels = pl.xticks()
# turn x labels
pl.setp(labels, rotation=45, size=20)
# Generate bar graph of number unique y's by class
_ = pl.bar(y_unique, counts)

# split into cross validation and normal
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# start pca
from sklearn.decomposition import RandomizedPCA
n_components = 150
print ("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = RandomizedPCA(n_components=n_components, whiten=True)
# test how long something takes
%time pca.fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))

# Replot data
eigenface_titles = [("eigenface %d" % i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

# Reshape X_train and X_test to new space
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Learn with SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', class_weight='auto')

# Cross validation to find good gamma, c on svm
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
cv = StratifiedShuffleSplit(y_train, test_size=0.20, n_iter=3)
# Use default C, gamma
%time svm_cv_scores = cross_val_score(svm, X_train_pca, y_train, scoring='f1', n_jobs=2)
svm_cv_scores

# Search for good C, gamma
from sklearn.grid_search import GridSearchCV
param_grid = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
clf = GridSearchCV(svm, param_grid, scoring='f1', cv=cv, n_jobs=2)
%time clf = clf.fit(X_train_pca, y_train)
# clf.best_params_ is best gamma, C

# Predict Y on test set
y_pred = clf.predict(X_test_pca)

# Plot data
prediction_titles = [title(y_pred, y_test, names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

# Check precision/recall etc
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=names))

# Confusion Matrix http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# Matrix of guesses by true values (same correct)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
print(cm)

