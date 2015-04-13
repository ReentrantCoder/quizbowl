from models import User
from random import shuffle
from numpy.ma.core import mean
import ast

from collections import Counter, defaultdict

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
            data.position = t["position"]
            train.append(data)
            
        test = []
        for t in self.testFormat.deserialize(testFilePath):
            questionId = t["question"]
            if questionId not in Q:
                continue

            data = TestData()
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
    
    def crossValidate(self, train, K, f):
        bucket = len(train)/K

        values = []
        for k in xrange(0, K):
            shuffle(train)
    
            asTest = train[:bucket]
            asTrain = train[bucket:]
            
            values.append( f(asTrain, asTest) )
        
        return mean(values)
    
    def groupByUser(self, (train, test)):
        U = {}

        for x in train:
            if not x.userId in U:
                U[x.userId] = User(x.userId);
            
            U[x.userId].train.append(x)
        
        for x in test:
            if not x.userId in U:
                U[x.userId] = User(x.userId);
            
            U[x.userId].test.append(x)

        return U

    def groupByCategory(self,(train,test)):
        C_train = {}
        C_test = {}
        U = self.groupByUser((train,test))
        #for each user determine the category for which they answer most questions
        #dictionary mapping category to users who answer most of the questions
        #return two dictionaries one for train and one for test
        for (id,user) in U.items():
            category_train = Counter()
            print user.train
            for q in user.train:
                category_train[q.questionCategory]+=1
            print category_train.most_common(1)
            if len(category_train)!=0:
                bestC = category_train.most_common(1)[0][0]
                try:
                    C_train[bestC].append(user)
                except KeyError:
                    C_train[bestC] = [user]
    
                category_test = Counter()
                for q in user.train:
                        category_test[q.questionCategory]+=1
                bestC = category_test.most_common(1)[0][0]
                try:
                    C_test[bestC].append(user)
                except KeyError:
                        C_test[bestC] = [user]
    
        return C_train,C_test

