from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from fileFormats import TrainFormat, TestFormat, QuestionFormat
from data import Dataset
from numpy.lib.function_base import average
from math import sqrt
from sklearn.linear_model.ridge import Ridge # works ok, but lasso better by few RMS pts
from sklearn.neighbors.classification import KNeighborsClassifier # not very accurate and slow
from sklearn.neighbors.regression import KNeighborsRegressor # not very accurate
from sklearn.svm.classes import SVR # Support vec machine was slow

class PositionPredictor:
    def __init__(self, trainingSet, alpha):
        self.vectorizer = CountVectorizer(analyzer=self, binary=True)
        self.model = Lasso(alpha=alpha)

        self.model.fit(
            self.__trainX(trainingSet), 
            self.__trainY(trainingSet)
            )

    def __call__(self, value):
        yield "C:" + value.questionCategory

#         yield "QID:" + str(value.questionId)
# 
#        yield "UID:" + str(value.userId)

#         for x in value.questionText[:100].split():
#             yield x
            
    def getPosition(self, testExample):
        return self.model.predict( self.__testX(testExample) )[0]

    def __testX(self, testExample):
        return self.vectorizer.transform([testExample])

    def __trainX(self, trainingSet):
        return self.vectorizer.fit_transform(trainingSet)
    
    def __trainY(self, trainingSet):
        return [abs(x.position) for x in trainingSet]


def toValidate(train, test, alpha):
    predictor = PositionPredictor(train, alpha)
    return sqrt(average(map(lambda (e, a): (e - a)*(e -a), [(abs(t.position), predictor.getPosition(t)) for t in test])))
            
if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

    baseAlpha = 10

    for i in xrange(0, 14):
        alpha = baseAlpha / pow(2.0, i)
        avgRMS, stdRMS = dataset.crossValidate(train, 5, lambda trainFold, testFold: toValidate(trainFold, testFold, alpha))
        print ("%f, %f, %f" % (alpha, avgRMS, stdRMS))