import pandas as pd
from sklearn import cross_validation, metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
import gc
import matplotlib
import matplotlib.pyplot as plt

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)

dfSum = pd.read_csv("F:\MSA\machine learning\project\Amusement Park\kk\\wzAfterCleaning.csv")
# dfSum = pd.read_csv("F:\MSA\machine learning\project\Amusement Park\kk\wzTestcleaned.csv")

# dfSum=dfSum.drop(['Unnamed: 0',"Weekday","StandardTemperature_t_1","Wind_t_1","StandardTemperature_t_2"], axis=1)

train_set, test_set = split_train_test(dfSum, 0.2)
# train_set=dfSum
# test_set= pd.read_csv("F:\MSA\machine learning\project\Amusement Park\kk\\df4_test.csv")

y=dfSum[['Ticket1', 'Ticket2']].as_matrix()
X = dfSum.drop(['Ticket1', 'Ticket2'], axis=1)
x = dfSum.drop(['Ticket1', 'Ticket2'], axis=1).as_matrix()
y_train = train_set[['Ticket1', 'Ticket2']].as_matrix()
y_train_1 = train_set[['Ticket1']].as_matrix()
y_train_2 = train_set[['Ticket2']].as_matrix()
x_train = train_set.drop(['Ticket1', 'Ticket2'], axis=1).as_matrix()
y_test = test_set[['Ticket1', 'Ticket2']].as_matrix()
y_test_1 = test_set[['Ticket1']].as_matrix()
y_test_2 = test_set[[ 'Ticket2']].as_matrix()
x_test = test_set.drop(['Ticket1', 'Ticket2'], axis=1).as_matrix()

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)
multi=MultiOutputRegressor(ada_clf)
multi.fit(x_train, y_train)
scores = cross_val_score(multi, x, y,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
# #('Mean:', 93.48802888231795)
# # ('Standard deviation:', 32.978215854285665)
#
# gc.enable()
# gc.collect()
ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=-1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)
multi=MultiOutputRegressor(ada_clf)
multi.fit(x_train, y_train)
scores = cross_val_score(multi, x, y,scoring="neg_mean_squared_error")
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
#
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
multi=MultiOutputRegressor(gbrt)
multi.fit(x_train, y_train)
scores = cross_val_score(multi, x, y,scoring="neg_mean_squared_error")
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
#
# ###randomforest
from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)
rf_random.best_estimator_

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

base_model = RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)
base_model.feature_importances_
# Model Performance
# Average Error: 14.3629 degrees.
# # Accuracy = 54.27%.
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)

# # Model Performance
# # Average Error: 13.2888 degrees.
# # Accuracy = 56.42%.
#
# #grid search with cross validation
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
#                           cv = 3, n_jobs = -1, verbose = 2)
# grid_search.fit(x_train, y_train)
# grid_search.best_params_
# # {'bootstrap': True,
# #  'max_depth': 90,
# #  'max_features': 3,
# #  'min_samples_leaf': 3,
# #  'min_samples_split': 8,
# #  'n_estimators': 100}
# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, x_test, y_test)
# # Average Error: 25.0963 degrees.
# # Accuracy = 4.98%.

# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=2, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=277, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)
# best_model=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=2, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=277, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)
# best_model.fit(x_train, y_train)
# best_accuracy = evaluate(best_model, x_test, y_test)
# best_model.feature_importances_

best_model=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=90,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=255, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

best_model.fit(x_train, y_train)
test_data=test_set.drop(["TimeStamp","TimeStamp_t_1","TimeStamp_t_2"],axis=1)
test_data=test_data.fillna(test_data.mean())
dfResult= pd.DataFrame(best_model.predict(test_data),columns=["1","2"])
test_data["1"]=dfResult["1"]
test_data["2"]=dfResult["2"]
test_data.to_csv('result_1.csv',index=False)

