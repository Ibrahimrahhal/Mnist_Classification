from classfier_metric import metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix

class Simple_Accurecy(metric):
    metricName = "Simple Accurecy"

    def metricFunction(self):
        return accuracy_score(self.trueValues, self.predValues)

class Balanced_Accurecy(metric):
    metricName = "Balanced Accurecy"

    def metricFunction(self):
        return balanced_accuracy_score(self.trueValues, self.predValues)

class Precision:
    metricName = "Precision"

    def metricFunction(self):
        return precision_score(self.trueValues, self.predValues)
         
class Recall:
    metricName = "Recall"

    def metricFunction(self):
        return recall_score(self.trueValues, self.predValues)

class F1_score:
    metricName = "F1_score"

    def metricFunction(self):
        return f1_score(self.trueValues, self.predValues)


class confusion_matrix:
    metricName = "confusion_matrix"

    def metricFunction(self):
        return confusion_matrix(self.trueValues, self.predValues)        


class Classification_Report:
    metricName = "classification_report"

    def metricFunction(self):
        return classification_report(self.trueValues, self.predValues)      


class Multi_Matrix:
    metricName = "multilabel_confusion_matrix"

    def metricFunction(self):
        return multilabel_confusion_matrix(self.trueValues, self.predValues)  


