import pandas as pd
# Dont use rows with NA in them
iris_data = pd.read_csv('iris-data.csv', na_values=['NA'])

# Look for missing data or anything odd
iris_data.describe()

# Plotting stuff
import matplotlib.pyplot as plt
import seaborn as sb
# Look for outliers, 0's, extra classes etc (dropna doesnt display null data which breaks plotting)
sb.pairplot(iris_data.dropna(), hue='class')

#### CLEANING DATA

# Fix extra classes, versicolor -> Iris-versicolor, Iris-setossa -> Iris-setosa
iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'
# Check to make sure unique classes are correct
iris_data['class'].unique()

# Only use entries with sepal width > 2.5
iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepal_width_cm'] >= 2.5)]
# plot to make sure outlier removed
iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepal_width_cm'].hist()

# look at near 0 entries for Iris-versicolor sepal length
iris_data.loc[(iris_data['class'] == 'Iris-versicolor') & (iris_data['sepal_length_cm'] < 1.0)]
# Convert accidental metres measurement to cm
iris_data.loc[(iris_data['class'] == 'Iris-versicolor') & (iris_data['sepal_length_cm'] < 1.0), 'sepal_length_cm'] *= 100.0
iris_data.loc[iris_data['class'] == 'Iris-versicolor', 'sepal_length_cm'].hist()

# Look at missing data entries (all setosa petal width missing)
iris_data.loc[(iris_data['sepal_length_cm'].isnull()) |
              (iris_data['sepal_width_cm'].isnull()) |
              (iris_data['petal_length_cm'].isnull()) |
              (iris_data['petal_width_cm'].isnull())]
# Get average setosa petal width
average_petal_width = iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].mean()
# Replace empty values with average petal width
iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
              (iris_data['petal_width_cm'].isnull()),
              'petal_width_cm'] = average_petal_width
# iris_data.dropna(inplace=True) could be used to drop empty rows

# Save cleaned data and reload
iris_data.to_csv('iris-data-clean.csv', index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')

#### TESTING
# Make sure 3 classes, min size for sepal 2.5cm and no 0 entries
assert len(iris_data_clean['class'].unique()) == 3
assert iris_data_clean.loc[iris_data_clean['class'] == 'Iris-versicolor', 'sepal_length_cm'].min() >= 2.5
assert len(iris_data_clean.loc[(iris_data_clean['sepal_length_cm'].isnull()) |
                               (iris_data_clean['sepal_width_cm'].isnull()) |
                               (iris_data_clean['petal_length_cm'].isnull()) |
                               (iris_data_clean['petal_width_cm'].isnull())]) == 0

#### CORRELATIONS
# Plot data not by class, look for collelations etc
sb.pairplot(iris_data_clean)

# Plot 'violin plots' nice looking box plots with smooth edges showing attribute distribution
plt.figure(figsize=(10, 10))
for column_index, column in enumerate(iris_data_clean.columns):
    if column == 'class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='class', y=column, data=iris_data_clean)

#### CLASSIFICATION
# Get X matrix
all_inputs = iris_data_clean[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values
# Get y matrix
all_classes = iris_data_clean['class'].values

# Split cross validation + test
from sklearn.cross_validation import train_test_split
(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75, random_state=1)

# Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(training_inputs, training_classes)
# Test how well model did
decision_tree_classifier.score(testing_inputs, testing_classes)

#Try fitting a bunch of times
model_accuracies = []
for repetition in range(1000):
    (training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)
    decision_tree_classifier.fit(training_inputs, training_classes)
    model_accuracies.append(decision_tree_classifier.score(testing_inputs, testing_classes))
sb.distplot(model_accuracies)

# K fold validation (split to k subsets, use each as test set once)
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
decision_tree_classifier = DecisionTreeClassifier()
# Get 10 cross validation scores
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)
sb.distplot(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))

#### TUNING
# Test depth = 1 (doesnt do very well)
decision_tree_classifier = DecisionTreeClassifier(max_depth=1)
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)
sb.distplot(cv_scores, kde=False)
plt.title('Average score: {}'.format(np.mean(cv_scores)))

# Grid search checks bunch of combinations of values, narrows its search and tries again for some number of times
from sklearn.grid_search import GridSearchCV
decision_tree_classifier = DecisionTreeClassifier()
parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}
# get cross validation sets
cross_validation = StratifiedKFold(all_classes, n_folds=10)
# Grid search of parameters
grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(all_inputs, all_classes)
# grid_search.best_score_ is score
# grid_search.best_params_ are best parameters

# Visualize accuracy by parameters
grid_visualization = []

for grid_pair in grid_search.grid_scores_:
    grid_visualization.append(grid_pair.mean_validation_score)
    
grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (5, 4)
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'][::-1])
plt.xlabel('max_features')
plt.ylabel('max_depth')

# Try diferent parameters including changing criteria and split
parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}

cross_validation = StratifiedKFold(all_classes, n_folds=10)

grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(all_inputs, all_classes)
# Set classifier to use best estimator
decision_tree_classifier = grid_search.best_estimator_

# Visualize decision tree
import sklearn.tree as tree
from sklearn.externals.six import StringIO
with open('iris_dtc.dot', 'w') as out_file:
    out_file = tree.export_graphviz(decision_tree_classifier, out_file=out_file)

# Random forest takes multiple decision trees, combines them when choosing classification
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
parameter_grid = {'n_estimators': [5, 10, 25, 50],
                  'criterion': ['gini', 'entropy'],
                  'max_features': [1, 2, 3, 4],
                  'warm_start': [True, False]}
cross_validation = StratifiedKFold(all_classes, n_folds=10)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(all_inputs, all_classes)
random_forest_classifier = grid_search.best_estimator_

# Compare performance of decision tree and random forest (plot classifiers results side by side)
rf_df = pd.DataFrame({'accuracy': cross_val_score(random_forest_classifier, all_inputs, all_classes, cv=10),
                       'classifier': ['Random Forest'] * 10})
dt_df = pd.DataFrame({'accuracy': cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10),
                      'classifier': ['Decision Tree'] * 10})
both_df = rf_df.append(dt_df)
sb.boxplot(x='classifier', y='accuracy', data=both_df)
sb.stripplot(x='classifier', y='accuracy', data=both_df, jitter=True, color='white')













