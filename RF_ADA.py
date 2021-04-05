import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import AdaBoostClassifier # Gradient Boosting (AdaBoost)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# read data from csv file into data-frame
df_train = pd.read_csv('lab4-train.csv')
df_test = pd.read_csv('lab4-test.csv')

# make training and testing sets
X_train = df_train.iloc[:, 0:4]
Y_train = df_train[['Class']]

X_test = df_test.iloc[:, 0:4]
Y_test = df_test[['Class']]

# make cross validation sets
kf_10 = KFold(n_splits=10, shuffle=True, random_state=1)


# === Random Forest ===

rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train, np.ravel(Y_train))
rf_10_cv = cross_val_score(rf_model, X_train, np.ravel(Y_train), cv=kf_10)
print("RF original model:", rf_model)

# grid search hyper-parameters
# rf_param_grid = {'min_samples_leaf': [1, 3, 5, 7, 9],
#                  'min_samples_split': [2, 3, 4, 5],
#                  'n_estimators': [10, 50, 100, 150, 200]}
# rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=10)
# rf_grid.fit(X_train, np.ravel(Y_train))
# print("RF best parameters:", rf_grid.best_params_)

best_rf = RandomForestClassifier(min_samples_leaf=9, min_samples_split=2, n_estimators=10, random_state=1)
print("RF best model:", best_rf)
best_rf.fit(X_train, np.ravel(Y_train))
print("RF cross_val_score:", cross_val_score(best_rf, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("RF accuracy on training data:", best_rf.score(X_train, Y_train))

scores_testing = best_rf.score(X_test, Y_test)
print("\nOverall classification accuracy of RF on testing data: %.2f%%" % (scores_testing*100))
confmat_rf = confusion_matrix(Y_test, best_rf.predict(X_test))
print('\nConfusion matrix of of RF on testing data:\n',
      confmat_rf)

# === ADA Boost ===

ada_model = AdaBoostClassifier(random_state=1)
ada_model.fit(X_train, np.ravel(Y_train))
ada_10_cv = cross_val_score(ada_model, X_train, np.ravel(Y_train), cv=kf_10)
print("ADA original model:", ada_model)

# grid search hyper-parameters
# ada_param_grid = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 2],
#                   'n_estimators': [10, 25, 50, 100, 200]}
# ada_grid = GridSearchCV(ada_model, ada_param_grid, cv=10)
# ada_grid.fit(X_train, np.ravel(Y_train))
# print("ADA best parameters:", ada_grid.best_params_)

best_ada = AdaBoostClassifier(learning_rate=0.5, n_estimators=100, random_state=1)
print("ADA best model:", best_ada)
best_ada.fit(X_train, np.ravel(Y_train))
print("ADA cross_val_score:", cross_val_score(best_ada, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("ADA accuracy on training data:", best_ada.score(X_train, Y_train))

scores_testing = best_ada.score(X_test, Y_test)
print("\nOverall classification accuracy of ADA on testing data: %.2f%%" % (scores_testing*100))
confmat_ada = confusion_matrix(Y_test, best_ada.predict(X_test))
print('\nConfusion matrix of of ADA on testing data:\n',
      confmat_ada)
