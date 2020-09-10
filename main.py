
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
from sklearn.preprocessing import label

mnist=fetch_openml('mnist_784',version=1,cache=True)
print(mnist)



X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)


import matplotlib
import matplotlib.pyplot as plt
some_digit = X[4]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
#plt.show()


X_train, X_test, y_train, y_test = X[:20000], X[20000:], y[:20000], y[20000:]
y_train = y_train.astype(np.int8)

import numpy as np
shuffle_index = np.random.permutation(20000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Training a Binary Classifier...
y_train_5 = (y_train ==5) # True for all 5s, False for all other digits.
y_test_5 = (y_test ==5)
#SGD classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr,label= None):
    plt.plot(fpr, tpr, linewidth=2,label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
plot_roc_curve(fpr, tpr,"SGD")
plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5, y_scores))
#Measuring Accuracy Using Cross-Validation
#from sklearn.model_selection import StratifiedKFold
#from sklearn.base import clone
#skfolds = StratifiedKFold(n_splits=3, random_state=42)
#for train_index, test_index in skfolds.split(X_train, y_train_5):
 #   clone = clone(sgd_clf)
  #  X_train_folds = X_train[train_index]
   # y_train_folds = (y_train_5[train_index])
    #X_test_fold = X_train[test_index]
    #y_test_fold = (y_train_5[test_index])
    #clone.fit(X_train_folds, y_train_folds)
    #y_pred = clone.predict(X_test_fold)
    #n_correct_pred = sum(y_pred == y_test_fold)
    #print(n_correct_pred / len(y_pred))


#Performance Measures of SGD
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


########################################

#train a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
(y_probas_forest)

#ROC CURVE
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend()
plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))

###########################################
#Multilabel Classification:KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
y_large = (y_train >= 7)
y_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_large, y_odd]
knn = KNeighborsClassifier()
knn.fit(X_train, y_multilabel)

print(knn.predict([some_digit]))


y_train_knn_pred = cross_val_predict(knn, X_train, y_train, cv=3)
print(f1_score(y_train, y_train_knn_pred, average="macro"))

#multiclass classification
#Let’s try this with the SGDClassifier:
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

#The highest score is indeed the one corresponding to class 9:
print(np.argmax(some_digit_scores))

print(sgd_clf.classes_)
print(sgd_clf.classes_[9])


#force ScikitLearn to use one-versus-one or one-versus-all:
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([some_digit]))

print(len(ovo_clf.estimators_))

#Random Forest classifiers can directly classify instances into multiple classes:
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

print(forest_clf.predict_proba([some_digit]))


