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
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer

class PositionPredictor:
    def __init__(self, trainingSet, alpha, userRatings, questionRatings):
        self.vectorizer = DictVectorizer()
#         self.vectorizer = CountVectorizer(analyzer=self, binary=True)
        self.model = Lasso(alpha)
        
        self.userRatings = userRatings
        self.questionRatings = questionRatings

        self.model.fit(
            self.__trainX(trainingSet), 
            self.__trainY(trainingSet)
            )

    def __call__(self, value):
        return {
#             "Category": value.questionCategory,
            "UserRating": self.userRatings[value.userId],
            "QuestionRating": self.questionRatings[value.questionId]
        }
 
#        yield "Category:" + value.questionCategory 
#         yield "UserRating:" + str(self.userRatings[value.userId])
#         yield "QuestionRating:" + str(self.questionRatings[value.questionId])



            
    def getPosition(self, testExample):
        return self.model.predict( self.__testX( testExample ) )

    def __testX(self, testExample):
        return self.vectorizer.transform( [self(x) for x in testExample] )
#         return self.vectorizer.transform( testExample )

    def __trainX(self, trainingSet):
        return self.vectorizer.fit_transform( [self(x) for x in trainingSet] )
#         return self.vectorizer.fit_transform( trainingSet )
    
    def __trainY(self, trainingSet):
        return [abs(x.position) for x in trainingSet]

def meanSquareError(predictions, test):
    return sqrt(average(map(lambda (a, b): (a - b)*(a - b), zip(predictions, [abs(x.position) for x in test]))))

def getPositionPredictions(dataset, train, test, alpha):
    byUser = dataset.groupByUser((train, test))
    userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 

    byQuestion = dataset.groupByQuestion((train, test))
    questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 

    predictor = PositionPredictor(train, alpha, userRatings, questionRatings)

    predictions = predictor.getPosition(test)
#     for i in xrange(0, len(predictions)):
#         predictions[i] -= 10
    
    return predictions

def reportPositionPrediction(dataset, training, alpha):
    print("Prediction MSE: %f +- %f" % 
          dataset.crossValidate(training, 10, lambda trainingFold, testFold: 
            meanSquareError(getPositionPredictions(dataset, trainingFold, testFold, alpha), testFold)))

if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    training, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    reportPositionPrediction(dataset, training, .1)


 
    byUser = dataset.groupByUser((training, test))
    userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 
 
    byQuestion = dataset.groupByQuestion((training, test))
    questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 
 
 
#     fTrain, fTest = dataset.splitTrainTest(training, len(training) / 5)
#     predictions = getPositionPredictions(dataset, fTrain, fTest, .1)
#     differences = map(lambda (x,y): x.position - y, zip(fTest, predictions))
#     for (x,y) in zip(fTest, differences):
#         print("%s, %s, %s, %f" % ( questionRatings[x.questionId] , x.questionCategory, userRatings[x.userId], x.position))
