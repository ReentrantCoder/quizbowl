from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from data import Dataset
from numpy.lib.function_base import average
from math import sqrt
from sklearn.linear_model.ridge import Ridge # works ok, but lasso better by few RMS pts
from sklearn.neighbors.classification import KNeighborsClassifier # not very accurate and slow
from sklearn.neighbors.regression import KNeighborsRegressor # not very accurate
from sklearn.svm.classes import SVR # Support vec machine was slow
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from numpy.core.umath import sign

class PositionPredictor:
    def __init__(self, dataset, trainingSet, alpha):
        byUser = dataset.groupByUser((trainingSet, None))
        byQuestion = dataset.groupByQuestion((trainingSet, None))

        self.userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 
        self.questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 

        self.vectorizer = DictVectorizer()
        self.model = Lasso(alpha)
        
        self.model.fit(
            self.__trainX(trainingSet), 
            self.__trainY(trainingSet)
            )

    def __call__(self, value):
        return {
            # Category does not contribute positively
            "Category": value.questionCategory
            ,
            "UserRating":  "NA" if not value.userId in self.userRatings else self.userRatings[value.userId]
#             ,
#             "QuestionRating": "NA" if not value.questionId in self.questionRatings else self.questionRatings[value.questionId]
            ,
            'LengthRating': round(len(value.questionText) / 38.0, 0) * 38
        }

    def getPosition(self, testExample):
        return self.model.predict( self.__testX( testExample ) )

    def __testX(self, testExample):
        return self.vectorizer.transform( [self(x) for x in testExample] )

    def __trainX(self, trainingSet):
        return self.vectorizer.fit_transform( [self(x) for x in trainingSet] )
    
    def __trainY(self, trainingSet):
        return [(x.position) for x in trainingSet]

def accuracy(predictions, test):
    return (sum( [ sign(p) * sign(t.position) for (p,t) in zip(predictions, test)  ] ) + len(test)) / (2.0 * len(test))

def meanSquareError(predictions, test):
    return sqrt(average(map(lambda (a, b): (a - b)*(a - b), zip(predictions, [ (x.position) for x in test]))))


def reportPositionPrediction(dataset, training, alpha):
    print("Prediction MSE: %f +- %f" % 
          dataset.crossValidate(training, 10, lambda trainingFold, testFold: 
            meanSquareError(PositionPredictor(dataset, trainingFold, alpha).getPosition(testFold), testFold)))

if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    training, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    fTraining, fTest = dataset.splitTrainTest(training, len(training)/5)
    fTraining, holdout = dataset.splitTrainTest(fTraining, len(fTraining) / 5)
    
    reportPositionPrediction(dataset, fTraining, .1)

    model = PositionPredictor(dataset, fTraining, .1)
    pTest = model.getPosition(fTest)
    pHoldout = model.getPosition(holdout)

    print meanSquareError(pTest, fTest)
    print accuracy(pTest, fTest)

    print meanSquareError(pHoldout, holdout)
    print accuracy(pHoldout, holdout)

#     for (p,f) in zip(pTest, fTest):
#         print str((f.position, p)).strip("(").strip(")")


#     guesses = [ {"id": t.id, "position": p} for (t, p) in zip(test, predictions)]
#     fileFormat = GuessFormat()
#     fileFormat.serialize(guesses , "data/guess.csv")