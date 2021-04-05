import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.linear_model import LogisticRegressionCV # Logistic Regression
from sklearn.naive_bayes import CategoricalNB # Naive Bayes
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import VotingClassifier # Majority Voting
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# read data from csv file into dataframe
df_train = pd.read_csv('lab4-train.csv')
df_test = pd.read_csv('lab4-test.csv')

# make training and testing sets
X_train = df_train.iloc[:, 0:4]
Y_train = df_train[['Class']]

X_test = df_test.iloc[:, 0:4]
Y_test = df_test[['Class']]

# make cross validation sets
kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)


# === Neural Network ===

nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train[:], np.ravel(Y_train))
nn_10_cv = cross_val_score(nn_model, X_train, np.ravel(Y_train), cv=kf_10)
print("NN original model:", nn_model)

# grid search hyper-parameters
nn_param_grid = {'hidden_layer_sizes': [(10, 10), (50, 50), (100, 100), (10, 20, 30),
                                        (20, 20, 20), (30, 20, 10), (50, 50, 50, 50)]}
nn_grid = GridSearchCV(nn_model, nn_param_grid, cv=10)
nn_grid.fit(X_train, np.ravel(Y_train))
print("NN best parameters:", nn_grid.best_params_)

best_nn = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42)
print("NN best model:", best_nn)
best_nn.fit(X_train, np.ravel(Y_train))
best_nn_probs = best_nn.predict_proba(X_train)[:, 1]
print("NN cross_val_score:", cross_val_score(best_nn, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("NN accuracy on training data:", best_nn.score(X_train, Y_train))

scores_testing = best_nn.score(X_test, Y_test)
print("\nOverall classification accuracy of NN on testing data: %.2f%%" % (scores_testing*100))
confmat_nn = confusion_matrix(Y_test, best_nn.predict(X_test))
print('\nConfusion matrix of of NN on testing data:\n',
      confmat_nn)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(confmat_nn, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat_nn.shape[0]):
#     for j in range(confmat_nn.shape[1]):
#         ax.text(x=j, y=i,
#             s=confmat_nn[i, j],
#             va='center', ha='center')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()


# === Logistic Regression ===

lr_model = LogisticRegressionCV(random_state=42)
lr_model.fit(X_train[:], np.ravel(Y_train))
lr_10_cv = cross_val_score(lr_model, X_train, np.ravel(Y_train), cv=kf_10)
print("LR original model:", lr_model)

# grid search hyper-parameters
lr_param_grid = {'Cs': [1, 10, 100]}
lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=10)
lr_grid.fit(X_train, np.ravel(Y_train))
print("LR best parameters:", lr_grid.best_params_)

best_lr = LogisticRegressionCV(Cs=1, random_state=42)
print("LR best model:", best_lr)
best_lr.fit(X_train, np.ravel(Y_train))
best_lr_probs = best_lr.predict_proba(X_train)[:, 1]
print("LR cross_val_score:", cross_val_score(best_lr, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("LR accuracy on training data:", best_lr.score(X_train, Y_train))

scores_testing = best_lr.score(X_test, Y_test)
print("\nOverall classification accuracy of LR on testing data: %.2f%%" % (scores_testing*100))
confmat_lr = confusion_matrix(Y_test, best_lr.predict(X_test))
print('\nConfusion matrix of of LR on testing data:\n',
      confmat_lr)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(confmat_lr, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat_lr.shape[0]):
#     for j in range(confmat_lr.shape[1]):
#         ax.text(x=j, y=i,
#             s=confmat_lr[i, j],
#             va='center', ha='center')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()


# === Naive Bayes ===

nb_model = GaussianNB()
nb_model.fit(X_train[:], np.ravel(Y_train))
nb_10_cv = cross_val_score(nb_model, X_train, np.ravel(Y_train), cv=kf_10)
print("NB original model:", nb_model)

best_nb = GaussianNB()
print("NB best model:", best_nb)
best_nb.fit(X_train, np.ravel(Y_train))
best_nb_probs = best_nb.predict_proba(X_train)[:, 1]
print("NB cross_val_score:", cross_val_score(best_nb, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("NB accuracy on training data:", best_nb.score(X_train, Y_train))

scores_testing = best_nb.score(X_test, Y_test)
print("\nOverall classification accuracy of NB on testing data: %.2f%%" % (scores_testing*100))
confmat_nb = confusion_matrix(Y_test, best_nb.predict(X_test))
print('\nConfusion matrix of of NB on testing data:\n',
      confmat_nb)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(confmat_nb, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat_nb.shape[0]):
#     for j in range(confmat_nb.shape[1]):
#         ax.text(x=j, y=i,
#             s=confmat_nb[i, j],
#             va='center', ha='center')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()


# === Decision Tree ===

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train[:], np.ravel(Y_train))
dt_10_cv = cross_val_score(dt_model, X_train, np.ravel(Y_train), cv=kf_10)
print(np.mean(dt_10_cv))
print(dt_10_cv.max())
print(dt_10_cv.min())
print(dt_10_cv.max()-dt_10_cv.min())
print("DT original model:", dt_model)

# grid search hyper-parameters
dt_param_grid = {'max_depth': [2, 3, 4, 5, 10, 100, None]}
dt_grid = GridSearchCV(dt_model, dt_param_grid, cv=10)
dt_grid.fit(X_train, np.ravel(Y_train))
print("DT best parameters:", dt_grid.best_params_)

best_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
print("DT best model:", best_dt)
best_dt.fit(X_train, np.ravel(Y_train))
best_dt_probs = best_dt.predict_proba(X_train)[:, 1]
print("DT cross_val_score:", cross_val_score(best_dt, X_train, np.ravel(Y_train), cv=kf_10).mean())
print("DT accuracy on training data:", best_dt.score(X_train, Y_train))

scores_testing = best_dt.score(X_test, Y_test)
print("\nOverall classification accuracy of DT on testing data: %.2f%%" % (scores_testing*100))
confmat_dt = confusion_matrix(Y_test, best_dt.predict(X_test))
print('\nConfusion matrix of of DT on testing data:\n',
      confmat_dt)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(confmat_dt, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat_dt.shape[0]):
#     for j in range(confmat_dt.shape[1]):
#         ax.text(x=j, y=i,
#             s=confmat_dt[i, j],
#             va='center', ha='center')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()


# === unweighted Voting Classifier ===

unweighted_voting_clf = VotingClassifier(
      estimators=[('nn', best_nn), ('lr', best_lr), ('nb', best_nb), ('dt', best_dt)],
      voting='soft')
unweighted_voting_clf.fit(X_train, np.ravel(Y_train))

print("\nunweighted majority vote results:\n")
for clf in (best_nn, best_lr, best_nb, best_dt, unweighted_voting_clf):
      clf.fit(X_train, np.ravel(Y_train))
      Y_pred = clf.predict(X_test)
      print(clf.__class__.__name__, accuracy_score(Y_test, Y_pred))


# === weighted hard Voting Classifier ===

# grid search hyper-parameters
weights = [1, 3, 2, 3]
# weights = [0.784, 0.814, 0.801, 0.814]

weighted_voting_clf = VotingClassifier(
      estimators=[('nn', best_nn), ('lr', best_lr), ('nb', best_nb), ('dt', best_dt)],
      voting='hard', weights=weights)
weighted_voting_clf.fit(X_train, np.ravel(Y_train))

print("\nweighted hard majority vote results:\n")
for clf in (best_nn, best_lr, best_nb, best_dt, weighted_voting_clf):
      clf.fit(X_train, np.ravel(Y_train))
      Y_pred = clf.predict(X_test)
      print(clf.__class__.__name__, accuracy_score(Y_test, Y_pred))


# === weighted soft Voting Classifier ===

soft_weighted_voting_clf = VotingClassifier(
      estimators=[('nn', best_nn), ('lr', best_lr), ('nb', best_nb), ('dt', best_dt)],
      voting='soft')
soft_weighted_voting_clf.fit(X_train, np.ravel(Y_train))

print("\nweighted soft majority vote results:\n")
for clf in (best_nn, best_lr, best_nb, best_dt, soft_weighted_voting_clf):
      clf.fit(X_train, np.ravel(Y_train))
      print(clf.__class__.__name__, clf.score(X_test, Y_test))