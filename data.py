from models import User

class TestData:
    def __init__(self):
        self.questionId = None
        self.questionCategory = None
        self.questionText = None
        self.questionAnswer = None
        self.userId = None

class TrainingData(TestData):
    def __init__(self):
        self.userAnswer = None
        self.position = None


class Dataset:
    def __init__(self, trainFormat, testFormat, questionFormat):
        self.trainFormat = trainFormat
        self.testFormat = testFormat
        self.questionFormat = questionFormat

    def getTrainingTest(self, trainFilePath, testFilePath, questionsFilePath):
        Q = self.questionFormat.generatorToDict(self.questionFormat.deserialize(questionsFilePath))
        
        train = []
        for t in self.trainFormat.deserialize(trainFilePath):
            questionId = t["question"]
            if questionId not in Q:
                continue
            
            data = TrainingData()
            data.questionId = questionId
            data.questionCategory = Q[data.questionId]["category"]
            data.questionText = Q[data.questionId]["question"]
            data.questionAnswer = Q[data.questionId]["answer"]
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
            data.questionId = t["question"]
            data.questionCategory = Q[data.questionId]["category"]
            data.questionText = Q[data.questionId]["question"]
            data.questionAnswer = Q[data.questionId]["answer"]
            data.userId = t["user"]
            test.append(data)
        
        return (train, test)
    
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