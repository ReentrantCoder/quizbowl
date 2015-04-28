from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from numpy.core.umath import sign
from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from numpy.lib.function_base import average
from math import sqrt

class PositionPredictor:
    def __init__(self, alpha):
        self.vectorizer = DictVectorizer()
        self.model = Lasso(alpha)

    def __call__(self, value):
        return {
            "Category": value.questionCategory
            ,
            "UserRating":  "NA" if not value.userId in self.userRatings else self.userRatings[value.userId]
            ,
            "QuestionRating": "NA" if not value.questionId in self.questionRatings else self.questionRatings[value.questionId]
            ,
            'LengthRating': round(len(value.questionText) / 38.0, 0) * 38
        }

    def fit(self, dataset, trainingSet):
        byUser = dataset.groupByUser((trainingSet, None))
        byQuestion = dataset.groupByQuestion((trainingSet, None))

        self.userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 
        self.questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 

        self.model.fit(
            self.__trainX(trainingSet), 
            self.__trainY(trainingSet)
            )
        
        return self.model

    def predict(self, testExample):
        return [ { "id": t.id, "position": p } for (p,t) in zip(self.model.predict( self.__testX( testExample ) ), testExample)  ]

    def __testX(self, testExample):
        return self.vectorizer.transform( [self(x) for x in testExample] )

    def __trainX(self, trainingSet):
        return self.vectorizer.fit_transform( [self(x) for x in trainingSet] )
    
    def __trainY(self, trainingSet):
        return [abs(x.position) for x in trainingSet]

class WordYesNoModel:
    def __call__(self, question):
        # User rating
        if question.userId in self.userRatings:
            yield "<U:" + str(self.userRatings[question.userId]) + ">"
        else:
            yield "<U:NA>"

        # Question rating
        if question.questionId in self.questionRatings:
            yield "<Q:" + str(self.questionRatings[question.questionId]) + ">"
        else:
            yield "<Q:NA>"

        # Length rating
        yield "<L:" + str( round(len(question.questionText)/40, 0)*40 ) + ">"

        if question.position:
            x = question.position
            # Leading words
            for word in question.questionText[:x].split()[-5:]:
                yield word
     
            # Trailing words
            for word in question.questionText[x:].split()[:5]:
                yield word
        else:
            for word in question.questionText[0:200].split():
                yield word



    def fit(self, train):
        wordsTrain = []
        Y = []

        byUser = dataset.groupByUser((train, None))
        byQuestion = dataset.groupByQuestion((train, None))

        self.userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 
        self.questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 

        for t in train:
            actual = round(t.position/5.0,0)*5.0
            wordsTrain.append( t )
            Y.append(1 if actual > 0 else -1)

        self.vectorizer = CountVectorizer(ngram_range=(1,1), binary=True, analyzer=self)
        X = self.vectorizer.fit_transform(wordsTrain)
    
        self.model = LogisticRegression()
        self.model.fit(X, Y)

    def predict(self, test):
        return self.model.predict(self.vectorizer.transform(test))

def meanSquareError(predictions, test):
    return sqrt(average(map(lambda (a,b): pow(a["position"] - (b.position), 2), zip(predictions, test))))

def meanAbsSquareError(predictions, test):
    return sqrt(average(map(lambda (a,b): pow(a["position"] - abs(b.position), 2), zip(predictions, test))))

def accuracy(predictions, test):
    return (len(predictions) + sum(map(lambda (a, b): sign(a["position"])*sign(b.position), zip(predictions, test)))) / float(2.0 * len(predictions))

def getSplitPredictions(fTrain, fTest):
    # Build a model based on all answers that will predict if the question will be answered right or wrong
    yesNoModel = WordYesNoModel()
    yesNoModel.fit(fTrain)
    yesNoPredict = yesNoModel.predict(fTest)
    print("yesNoModel ACC: %f" % (( sum([ p * sign(t.position) for (p, t) in zip(yesNoPredict, fTest)]) + len(fTest) )/ (2.0 * len(fTest))))

    # Split training into two sets: all correct answers, and all negative answers
    byCorrectness = dataset.groupBy((fTrain, fTest), lambda x: x.isCorrect)

    # Build a model based on all correct answers that will predicted the position 
    yesPosModel = PositionPredictor(0.1)
    yesPosModel.fit(dataset, byCorrectness[True].train)
    yesPredict = yesPosModel.predict( fTest )
    yesLookup = { t.id : p for (p, t) in zip(yesPredict, fTest)  }
    print("yesPosModel MSE: %f" % meanAbsSquareError(yesPredict, fTest))

    # Build a model based on all incorrect answers that will predict the abs(position)
    noPosModel = PositionPredictor(.1)
    noPosModel.fit(dataset, byCorrectness[False].train)
    noPredict = yesPosModel.predict( fTest )
    noLookup = { t.id : p for (p, t) in zip(noPredict, fTest)  }
    print("noPosModel MSE: %f" % meanAbsSquareError(noPredict, fTest ))
    print("")

    predictions = []
    for (p, t) in zip(yesNoPredict, fTest):
        if p > 0:
            predictions.append( { "id": t.id, "position": yesLookup[t.id]["position"] }  )
        else:
            predictions.append( { "id": t.id, "position": -noLookup[t.id]["position"] }  )

    return predictions

def getPredictions(train, test):
    # Build a model based on all answers that will predict if the question will be answered right or wrong
    yesNoModel = WordYesNoModel()
    yesNoModel.fit(train)
    yesNoPredict = yesNoModel.predict(test)

    # Split training into two sets: all correct answers, and all negative answers
    byCorrectness = dataset.groupBy((train, None), lambda x: x.isCorrect)

    # Build a model based on all correct answers that will predicted the position 
    yesPosModel = PositionPredictor(0.1)
    yesPosModel.fit(dataset, byCorrectness[True].train)
    yesPredict = yesPosModel.predict( test )
    yesLookup = { t.id : p for (p, t) in zip(yesPredict, test)  }

    # Build a model based on all incorrect answers that will predict the abs(position)
    noPosModel = PositionPredictor(.1)
    noPosModel.fit(dataset, byCorrectness[False].train)
    noPredict = yesPosModel.predict( test )
    noLookup = { t.id : p for (p, t) in zip(noPredict, test)  }

    predictions = []
    for (p, t) in zip(yesNoPredict, test):
        if p > 0:
            predictions.append( { "id": t.id, "position": yesLookup[t.id]["position"] }  )
        else:
            predictions.append( { "id": t.id, "position": -noLookup[t.id]["position"] }  )

    return predictions

if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

#     print("%f %f" % dataset.crossValidate(train, 5, lambda trainFold, testFold: 
#                                           meanSquareError(getSplirtPredictions(trainFold, testFold), testFold)))

    fTrain, fTest = dataset.splitTrainTest(train, len(train)/5)
    fPredictions = getPredictions(fTrain, fTest)
    print("combined MSE: %f" % meanSquareError(fPredictions, fTest) )
    print("combined ACC: %f" % accuracy(fPredictions, fTest))

#     for (p, t) in zip(fPredictions, fTest):
#         print str((t.questionCategory, len(t.questionText), t.position, p["position"])).strip("(").strip(")")
        
#     predictions = getPredictions(train, test)
#     guessFormat = GuessFormat()
#     guessFormat.serialize(predictions, "data/guess.csv")

    # Estimate Kaggle score based on intersection of test and train
    predictions = getPredictions(train, test)

    A = set([t.questionId for t in train])
    B = set([t.questionId for t in test])
    cap = A.intersection(B)
    
    trainLookup = {t.questionId : t for t in train}
    predictionLookup = { t.questionId : p for (t, p) in zip(test, predictions) }

    T = []
    P = []
    for questionId in cap:
        T.append(trainLookup[questionId]) 
        P.append(predictionLookup[questionId])
    
    print("overlap MSE: %f" % meanSquareError(P, T) )
    print("overlap ACC: %f" % accuracy(P, T))