from models import User
from random import shuffle
from numpy.ma.core import mean
import ast

from collections import Counter, defaultdict
from math import sqrt


class TestData:
    def __init__(self):
        self.id = None
        self.questionId = None
        self.questionCategory = None
        self.questionText = None
        #self.questionWords = None
        self.questionAnswer = None
        self.userId = None

    def getQuestionFragment(self):
        questionFragment = ""
        if self.position == None or self.position <= 0:
            return questionFragment
        
        if self.position >= len(self.questionText):
            return self.questionText
        
        return self.questionText[:self.position]

class TrainingData(TestData):
    def __init__(self):
        self.userAnswer = None
        self.position = None


class Dataset:
    def __init__(self, trainFormat, testFormat, questionFormat):
        self.trainFormat = trainFormat
        self.testFormat = testFormat
        self.questionFormat = questionFormat

    def getTrainingTest(self, trainFilePath, testFilePath, questionsFilePath, limit):
        Q = self.questionFormat.generatorToDict(self.questionFormat.deserialize(questionsFilePath))
        
        train = []
        for t in self.trainFormat.deserialize(trainFilePath):
            questionId = t["question"]
            if questionId not in Q:
                continue
            
            data = TrainingData()
            data.id = t["id"]
            data.questionId = questionId
            data.questionCategory = Q[data.questionId]["category"]
            data.questionText = Q[data.questionId]["question"]
            data.questionAnswer = Q[data.questionId]["answer"]
            #data.questionWords = ast.literal_eval(Q[data.questionId]["blob"])
            data.userId = t["user"]
            data.userAnswer = t["answer"]
            data.position = int(float(t["position"]))
            data.isCorrect = (data.position > 0)
            train.append(data)
            
        test = []
        for t in self.testFormat.deserialize(testFilePath):
            questionId = t["question"]
            if questionId not in Q:
                continue

            data = TrainingData()
            data.id = t["id"]
            data.questionId = t["question"]
            data.questionCategory = Q[data.questionId]["category"]
            data.questionText = Q[data.questionId]["question"]
            data.questionAnswer = Q[data.questionId]["answer"]
            #data.questionWords = ast.literal_eval(Q[data.questionId]["blob"])
            data.userId = t["user"]
            test.append(data)
        
        if limit < 0:
            return (train, test)
        
        return (train[:limit], test[:limit])

    def splitTrainTest(self, train, bucket):
        shuffle(train)
        asTest = train[:bucket]
        asTrain = train[bucket:]
        return asTrain, asTest
    
    def crossValidate(self, train, K, f):
        bucket = len(train)/K

        values = []
        for k in xrange(0, K):
            asTrain, asTest = self.splitTrainTest(train, bucket)
            values.append( f(asTrain, asTest) )

        mu = mean(values)
        sigma = sqrt(mean(mean([ pow(x - mu, 2) for x in values])))
                
        return (mu, sigma)
    
    def groupBy(self, (train, test), f):
        U = {}

        for x in train:
            _key = f(x)
            if not _key in U:
                U[_key] = User(_key);
            
            U[_key].train.append(x)
        
        if test != None:
            for x in test:
                _key = f(x)
                if not _key in U:
                    U[_key] = User(_key);
                
                U[_key].test.append(x)

        return U

    def groupByUser(self, (train, test)):
        return self.groupBy((train, test), lambda x: x.userId)

    def groupByQuestion(self, (train, test)):
        return self.groupBy((train, test), lambda x: x.questionId)

    def groupByCategory(self,(train,test)):
        return self.groupBy((train, test), lambda x: x.questionCategory)
