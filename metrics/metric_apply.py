from metrics.classification import Simple_Accurecy, Balanced_Accurecy, Precision, Recall, F1_score, confusion_matrix, Classification_Report, Multi_Matrix
import seaborn as sns
import numpy as np
def applyForBinaryClassification(TrueData, PredData):
    print(Simple_Accurecy(TrueData, PredData).calc())
    print(Balanced_Accurecy(TrueData, PredData).calc())
    print(Precision(TrueData, PredData).calc())
    print(Recall(TrueData, PredData).calc())
    print(F1_score(TrueData, PredData).calc())
    print(confusion_matrix(TrueData, PredData).calc())



def applyForMultiClassification(TrueData, PredData):
    print(Simple_Accurecy(TrueData, PredData).calc())
    print(Balanced_Accurecy(TrueData, PredData).calc())
    print(Classification_Report(TrueData, PredData).calc())
    print(Multi_Matrix(TrueData, PredData).calc())


def matrixVisualization(TrueData, PredData , isMulti = False):
    matrix = Multi_Matrix(TrueData, PredData).getCalc() if isMulti else confusion_matrix(TrueData, PredData).getCalc() 
    sns.heatmap(matrix/np.sum(matrix), annot=True, fmt='.2%', cmap='Blues')


