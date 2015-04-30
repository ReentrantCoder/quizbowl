from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from numpy.core.umath import sign
from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from numpy.lib.function_base import average
from math import sqrt
from random import random

class PositionPredictor:
    def __init__(self, alpha, userGranularity, questionGranularity):
        self.userGranularity = userGranularity
        self.questionRatings = questionGranularity
        
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

        self.userRatings = { userId : user.getRating(self.userGranularity)  for (userId, user) in byUser.items()  } 
        self.questionRatings = { questionId : question.getRating(self.questionRatings)  for (questionId, question) in byQuestion.items()  } 

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
    def __init__(self, userGranularity, questionGranularity):
        self.userGranularity = userGranularity
        self.questionRatings = questionGranularity
    
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

#         if question.position:
#             x = question.position
#             # Leading words
#             for word in question.questionText[:x].split()[-5:]:
#                 yield word
#      
#             # Trailing words
#             for word in question.questionText[x:].split()[:5]:
#                 yield word
#         else:
        x = int(0.12 * len(question.questionText))
        words = question.questionText[0:x].split()
        for (next, curr) in zip(words[1:], words):
            if(curr.lower() in ["this", "these", "that", "those"]):
                yield next

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
    return sqrt(average(map(lambda (a,b): pow(abs(a["position"]) - abs(b.position), 2), zip(predictions, test))))

def accuracy(predictions, test):
    return (len(predictions) + sum(map(lambda (a, b): sign(a["position"])*sign(b.position), zip(predictions, test)))) / float(2.0 * len(predictions))

def accuracySign(predictions, test):
    return (len(predictions) + sum(map(lambda (a, b): a*sign(b.position), zip(predictions, test)))) / float(2.0 * len(predictions))


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
    yesNoModel = WordYesNoModel(10, 10)
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

class PosCorrPredictor:
    def __init__(self, userGranularity, questionGranularity):
        self.userGranularity = userGranularity 
        self.questionGranularity = questionGranularity 

        self.vectorizer = DictVectorizer()
        self.model = LogisticRegression()
    
    def fit(self, dataset, training):
        byUser = dataset.groupByUser((train, None))
        byQuestion = dataset.groupByQuestion((train, None))

        self.userRatings = { userId : user.getRating(self.userGranularity)  for (userId, user) in byUser.items()  } 
        self.questionRatings = { questionId : question.getRating(self.questionGranularity)  for (questionId, question) in byQuestion.items()  } 

        X = self.trainX(training)
        X = self.vectorizer.fit_transform(X)
        Y = self.trainY(training)
        
        self.model.fit(X, Y)
        
        return self

    def getFeature(self, text, x):
        count = 0
        x = int(round(x))+1
        words = text[0:x].split()
        for (prev, next) in zip(words, words[1:]):
#             #Capture the number of instances where the subject is mentioned
#             if prev.lower().strip() in ["this", "that", "these", "those"]:
#                 count += 1

            # Attempt to capture the number of hints which are assumed to be separated by ", . ; and or"
            if prev.endswith(",") or prev.endswith(".") or prev.endswith(";") or prev.lower() in ["and", "or"]:
                count += 1
        
        return count

    def predict(self, test, predictedPositions):
        return self.model.predict(self.vectorizer.transform(self.testX(test, predictedPositions)))
        
    def predict_proba(self, test, predictedPositions):
        return self.model.predict_proba(self.vectorizer.transform(self.testX(test, predictedPositions)))

    def testX(self, test, predictedPositions):
        X = []
        for (t,p) in zip(test, predictedPositions):
            X.append({
                "userRating": "NA" if not t.userId in self.userRatings else self.userRatings[t.userId],
                "questionRating": "NA" if not t.questionId in self.questionRatings else self.questionRatings[t.questionId],
#                 "lengthRating": str( round(len(t.questionText)/40, 0)*40 ),
#                 "textHints": self.getFeature(t.questionText, p["position"])
            })

        return X

    def trainX(self, training):
        X = []
        for t in training:
            X.append({
                "userRating": "NA" if not t.userId in self.userRatings else self.userRatings[t.userId],
                "questionRating": "NA" if not t.questionId in self.questionRatings else self.questionRatings[t.questionId],
#                 "lengthRating": str( round(len(t.questionText)/40, 0)*40 ),
#                 "textHints": self.getFeature(t.questionText, t.position)
            })

        return X

    def trainY(self, training):
        Y = []
        for t in training:
            if t.position > 0:
                Y.append(1)
            else:
                Y.append(-1)

        return Y

def getPosToCorrPredictions(dataset, trainFold, testFold):
    posPredictor = PositionPredictor(1/20.0, 7, 10)
    posPredictor.fit(dataset, trainFold)
    positionPredictions = posPredictor.predict(testFold)

    corPredictor = PosCorrPredictor(12, 9)
    corPredictor.fit(dataset, trainFold)
    yesNoPredictions = corPredictor.predict(testFold, positionPredictions)
    yesNoProbs = corPredictor.predict_proba(testFold, positionPredictions)
 
    return [ {"id": p["id"], "position": - 0.246850394 * p["position"] if c < 0 else 0.609901599 * p["position"] } for (p, c) in zip(positionPredictions, yesNoPredictions) ]
    
    #return [ {"id": p["id"], "position": - 0.246850394 * hedge_bets(p["position"], conf) if c < 0 else 0.609901599 * hedge_bets(p["position"], conf) } for (p, c, conf) in zip(positionPredictions, yesNoPredictions, yesNoProbs) ]

def hedge_bets(guess, conf, mean = 39.298062750052644):
    ## diff is between 0 (no confidence) and 1 (high confidence)
    ## it is the difference in probabilities for predicting yes and no, respectively
    ## output is confidence-weight position guess
    ## the less confident the guess, the more it is dragged towards the mean
    diff = abs(conf[0] - conf[1]) 
    return guess*diff + mean*(1 - diff)

if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

    # Figure out what percent of users are in both train and test sets
    A = set([ t.userId for t in train ])
    B = set([ t.userId for t in test ])
    cap = A.intersection(B)
    cup = A.union(B)
    knownUsers = len(cap) / float(len(cup))

    # Figure out what percent of questions are in both train and test sets
    A = set([ t.questionId for t in train ])
    B = set([ t.questionId for t in test ])
    cap = A.intersection(B)
    cup = A.union(B)
    knownQuestions = len(cap) / float(len(cup))

    # Make the training set look like the test set by introducing uncertainty
    for t in train:
        if( random() > knownUsers ):
            t.userId = "NA"
        if( random() > knownQuestions ):
            t.questionId = "NA"

    print("%f %f" % dataset.crossValidate(train, 10, lambda trainFold, testFold: 
        meanSquareError(getPosToCorrPredictions(dataset, trainFold, testFold), testFold)))
# 
#     fTrain, fTest = dataset.splitTrainTest(train, len(train)/5)
#     predictions = getPosToCorrPredictions(dataset, fTrain, fTest)
#     for (p, t) in zip(predictions, fTest):
#         print str((t.questionCategory, len(t.questionText), t.position, p["position"])).strip("(").strip(")")

#     predictions = getPosToCorrPredictions(dataset, train, test)
#     guessFormat = GuessFormat()
#     guessFormat.serialize(predictions, "data/guess.csv")
