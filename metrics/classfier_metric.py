from abc import ABC, abstractmethod
class metric(ABC):
    
    metricName = 'default'

    def __init__(true, pred, desc, self):
        self.trueValues = true
        self.predValues = pred
        self.desc = desc

    def calc(self):
        return self.desc + " With " + self.metricName + " :- " +self.metricFunction(self.trueValues, self.predValues)

    def getCalc(self):
        return self.metricFunction(self.trueValues, self.predValues)

    @property
    @abstractmethod
    def metricFunction(self):
        pass
