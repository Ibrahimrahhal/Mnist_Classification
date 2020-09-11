from abc import ABC, abstractmethod
class metric(ABC):
    
    metricName = 'default'

    def __init__(self, true, pred, desc="" ):
        self.trueValues = true
        self.predValues = pred
        self.desc = desc

    def calc(self):
        if(metric != 'classification_report'):
            return self.desc + " With " + self.metricName + " :- " + str(self.metricFunction())
        else:
            return self.desc + " With " + self.metricName + " :- \n" + str(self.metricFunction())   

    def getCalc(self):
        return self.metricFunction()

    @property
    @abstractmethod
    def metricFunction(self):
        pass
