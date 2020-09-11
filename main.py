from dataset.mnist import load_ready_mnist
from metrics.metric_apply import applyForBinaryClassification, applyForMultiClassification, matrixVisualization

import matplotlib as matplotlib
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pip._vendor.webencodings import labels

from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_curve, plot_roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_predict
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


X_train, X_test, y_train, y_test = load_ready_mnist()

#Training a Binary Classifier...
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

print(y_train.shape)
#SGD classifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
applyForBinaryClassification(y_test_5, sgd_clf.predict(X_test), "SGD classifier ")



#multiclass classification
#Let’s try this with the SGDClassifier:
sgd_clf_multi = SGDClassifier(random_state=42)
sgd_clf_multi.fit(X_train, y_train) # y_train, not y_train_5
applyForMultiClassification(y_test, sgd_clf_multi.predict(X_test), "SGD classifier Multi")

#Multilabel Classification:KNeighborsClassifier
y_large = (y_train >= 7)
y_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_large, y_odd]
knn = KNeighborsClassifier()
knn.fit(X_train, y_multilabel)
applyForMultiClassification(np.c_[(y_test >= 7), (y_test % 2 == 1)], knn.predict(X_test), "KNeighborsClassifier")

#force ScikitLearn t o use one-versus-one or one-versus-all:

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
applyForMultiClassification(y_test, ovo_clf.predict(X_test), "OneVsOneClassifier")

#Random Forest classifiers can directly classify instances into multiple classes:
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
applyForMultiClassification(y_test, forest_clf.predict(X_test), "Random Forest")


# #Measuring Accuracy Using Cross-Validation
# #from sklearn.model_selection import StratifiedKFold
# #from sklearn.base import clone
# #skfolds = StratifiedKFold(n_splits=3, random_state=42)
# #for train_index, test_index in skfolds.split(X_train, y_train_5):
#  #   clone = clone(sgd_clf)
#   #  X_train_folds = X_train[train_index]
#    # y_train_folds = (y_train_5[train_index])
#     #X_test_fold = X_train[test_index]
#     #y_test_fold = (y_train_5[test_index])
#     #clone.fit(X_train_folds, y_train_folds)
#     #y_pred = clone.predict(X_test_fold)
#     #n_correct_pred = sum(y_pred == y_test_fold)
#     #print(n_correct_pred / len(y_pred))


# #Performance Measures of SGD
# from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# ########################################

# #train a RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
# print(y_probas_forest)


# #y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
# #fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)



# ###########################################


# print(knn.predict([some_digit]))


# y_train_knn_pred = cross_val_predict(knn, X_train, y_train, cv=3)
# print(f1_score(y_train, y_train_knn_pred, average="macro"))





# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)

# #The highest score is indeed the one corresponding to class 9:
# print(np.argmax(some_digit_scores))

# print(sgd_clf.classes_)
# print(sgd_clf.classes_[9])



# print(len(ovo_clf.estimators_))



# print(forest_clf.predict_proba([some_digit]))


