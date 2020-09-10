from metrics.classification import Simple_Accurecy, Balanced_Accurecy, Precision, Recall, F1_score, confusion_matrix, Classification_Report, Multi_Matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix;
def matrixVisualization(classifier, X_test, y_test):
    disp = plot_confusion_matrix(classifier, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()


def applyForBinaryClassification(TrueData, PredData, desc):
    print(Simple_Accurecy(TrueData, PredData, desc).calc())
    print(Balanced_Accurecy(TrueData, PredData, desc).calc())
    print(Precision(TrueData, PredData, desc).calc())
    print(Recall(TrueData, PredData, desc).calc())
    print(F1_score(TrueData, PredData, desc).calc())
    # matrixVisualization(TrueData, PredData, False)


def applyForMultiClassification(TrueData, PredData, desc):
    print(Classification_Report(TrueData, PredData, desc).calc())
    print(Multi_Matrix(TrueData, PredData, desc).getCalc())




