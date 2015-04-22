from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from data import Dataset
from fileFormats import TrainFormat, QuestionFormat, TestFormat
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer

class Analyzer:
    def __init__(self, useQuestionId, useUserId, useCategory, useQuestionText, userRatings, questionRatings, useNumCount):
        self.useQuestionId = useQuestionId
        self.useUserId = useUserId
        self.useCategory = useCategory
        self.useQuestionText = useQuestionText

        self.useNumCount = useNumCount

        self.userRatings = userRatings
        self.questionRatings = questionRatings
    
    def __call__(self, example):
        if self.useQuestionId:
            yield "Q:" + str(example.questionId)
            
        if self.useUserId:
            yield "U:" + str(example.userId)
            
        if self.useCategory:
            yield "C:" + example.questionCategory
        
        if self.userRatings != None:
            if example.userId in self.userRatings:
                yield "UR:" + str(self.userRatings[example.userId])

        if self.questionRatings != None:
            if example.questionId in self.questionRatings:
                yield "QR:" + str(self.questionRatings[example.questionId])
            
        if self.useNumCount:
            areNumbers = filter(lambda x: self.__isNumber(x), example.questionText.split())
            yield "NC:" + str(len(areNumbers))

    def __isNumber(self, word):
        try:
            float(word)
            return True
        except ValueError:
            return False
            
class Featurizer:
    def __init__(self, analyzer):
        self.vectorizer = CountVectorizer(analyzer=analyzer, binary=True)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)


# class DictAnalyzer:
#     def __init__(self, useQuestionId, useUserId, useCategory, useQuestionText, userRatings, questionRatings, useNumCount):
#         self.useQuestionId = useQuestionId
#         self.useUserId = useUserId
#         self.useCategory = useCategory
#         self.useQuestionText = useQuestionText
# 
#         self.useNumCount = useNumCount
# 
#         self.userRatings = userRatings
#         self.questionRatings = questionRatings
#     
#     def __call__(self, example):
#         A = {}
#         
#         if self.useQuestionId:
#             A["QuestionId"] = example.questionId
#             
#         if self.useUserId:
#             A["UserId"] = example.userId
#             
#         if self.useCategory:
#             A["Category"] = example.questionCategory
#         
#         if self.userRatings != None:
#             if example.userId in self.userRatings:
#                 A["UserRating"] = self.userRatings[example.userId]
# 
#         if self.questionRatings != None:
#             if example.questionId in self.questionRatings:
#                 A["QuestionRating"] = self.questionRatings[example.questionId]
#             
#         if self.useNumCount != None:
#             wordLengths = 0
#             words = example.questionText.split()
#             for x in words:
#                 if(self.__isNumber(x)):
#                     self.__increment(A, "Numbers")
# 
#                 if x[0].isupper():
#                     self.__increment(A, "Propers")
#                 
#                 wordLengths += len(x)
#             
#             A["AvgWordLen"] = round(wordLengths / float(len(words)))
#             
#         return A
# 
#     def __increment(self, A, key):
#         if key not in A:
#             A[key] = 1
#         else:
#             A[key] += 1
# 
#     def __isNumber(self, word):
#         try:
#             float(word)
#             return True
#         except ValueError:
#             return False
# 
# class DictFeaturizer:
#     def __init__(self, analyzer):
#         self.analyzer = analyzer
#         self.vectorizer = DictVectorizer()
# 
#     def train_feature(self, examples):
#         return self.vectorizer.fit_transform( [ self.analyzer(x) for x in examples ] )
# 
#     def test_feature(self, examples):
#         return self.vectorizer.transform( [ self.analyzer(x) for x in examples ]  )
#     
class CorrectnessPredictor:
    def __init__(self, featurizer, trainingData):
        self.featurizer = featurizer
         
        self.model = SGDClassifier('log', penalty='l2', shuffle=True)
        self.model.fit(
           featurizer.train_feature(trainingData),
           [x.isCorrect for x in trainingData]
           )
 
    def predict(self, testData):
        return self.model.predict(self.featurizer.test_feature(testData))

def getCorrectnessAnalyzer(dataset, training, test):
    byUser = dataset.groupByUser((training, test))
    userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 

    byQuestion = dataset.groupByQuestion((training, test))
    questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  } 

    return Analyzer(
         useQuestionId = True, useUserId = True, useCategory = True, useQuestionText = False, 
         userRatings = userRatings, questionRatings = questionRatings,
         useNumCount = False
         )

def getCorrectnessPredictions(dataset, training, test):
    predictor = CorrectnessPredictor(Featurizer(getCorrectnessAnalyzer(dataset, training, test)), training)
    
    return predictor.predict(test)

def accuracy(predictions, test):
    return sum(map(lambda (a, b): a == b, zip(predictions, [x.isCorrect for x in test]))) / float(len(test))
           
def reportCorrectness(dataset, training):
    print("Prediction Accuracy: %f +- %f" % dataset.crossValidate(training, 10, 
        lambda trainingFold, testFold: 
            accuracy(getCorrectnessPredictions(dataset, trainingFold, testFold), testFold)))
    
if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    training, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    reportCorrectness(dataset, training)
