from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from math import sqrt
from numpy.lib.function_base import average
from numpy.core.umath import sign
from copy import deepcopy
from scipy.stats.stats import pearsonr
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from combinedApproach import PositionPredictor
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.linear_model.ridge import Ridge
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from ctypes.test.test_prototypes import positive_address

class RatingModel:
    def __init__(self, userGranularity, questionGranularity):
        self.userGranularity = userGranularity
        self.questionGranularity = questionGranularity

        self.adjustments = { -130: 112.5, -120: 170, -110: 8, -100: 88.5714285714286, -90: 108.2, -80: 79.9009900990099, -70: 72.0338983050847, -60: -20, -50: 58, -40: 30, -30: 13.8461538461538, -20: 120, -10: 22.6966292134831, 0: 34.9068322981366, 10: -0.963855421686747, 20: 14.2105263157895, 30: 10.5316455696203, 40: 10.968992248062, 50: 0.740740740740741, 60: -18.9328063241107, 70: -19.879781420765, 80: -15.2830188679245, 90: -82.2222222222222, 130: -240 }
        self.adjustments2 = {-140: 25, -110: 5, -100: 110, -80: 80, -20: -0.72463768115942, -10: 9.30232558139535, 0: -1.06060606060606, 10: -14.7717842323651, 20: -22.8235294117647, 30: -0.766423357664234, 40: 2.86971830985915, 50: -2.41590214067278, 60: -7.32638888888889, 70: -4, 100: -84.0384615384615}
        self.adjustments3 = { -110: 0, -100: 140, -80: 58.3333333333333, -20: -5.65217391304348, -10: -11.5151515151515, 0: -5.85551330798479, 10: 11.8518518518519, 20: -6.19354838709677, 30: -3.24444444444444, 40: 0.793357933579336, 50: 0.719114935464044, 60: 11.9012345679012, 70: -30, 100: -108.552631578947 }
        self.adjustments4 = { -180: 130, -30: -25.2941176470588, -20: 32.7027027027027, -10: 15.8629441624365, 0: -6.71875, 10: -15, 20: -24.6875, 30: -0.405797101449275, 40: -0.257966616084977, 50: -2.82174810736407, 60: -30, 70: 2.33516483516484, 80: -2.5 }
 
        self.qRatingAdjustments = {
                                   -1: { -10: -10.5, 0: -62.5, 10: 3.4, 20: -83.3, 30:-20},
                                   0: {-160:50, -50:28.1, -10:-4, 0:-51.3, 10:-24.0, 20:-6.7, 30:-3, 40:3.7, 50:-15},
                                   1: { -10:25, 0:110, 10:-29.3,40:6.6,70:26.7,80:-18.3 },
                                   "NA": {50:-19.8}
                                   }

    def fit(self, dataset, train):
        byUser = dataset.groupByUser((train, None))
        self.userRatings = { userId : user.getRating(self.userGranularity)  for (userId, user) in byUser.items()  } 
    
        byQuestion = dataset.groupByQuestion((train, None))
        self.questionRatings = { questionId : question.getRating(self.questionGranularity)  for (questionId, question) in byQuestion.items()  }
        
        # Train
        self.expectedPosition = {}
        for t in train:
            categorySlice = self.getSlice(self.expectedPosition, t.questionCategory, {})
            
            userRating = self.userRatings[t.userId]
            userSlice = self.getSlice(categorySlice, userRating, {})
    
            questionRating = self.questionRatings[t.questionId]
            (sum, N) = self.getSlice(userSlice, questionRating, (0.0, 0.0))
                
            userSlice[questionRating] = (sum + t.position, N + 1.0)
    
        self.averagePosition = self.getAveragePosition(self.expectedPosition)
        self.averageCategoryPosition = self.getAverageCategoryPosition(self.expectedPosition)
        self.avergaeUserCategoryPosition = self.getAvgPositionByUserRatingAndCategory(self.expectedPosition)

        return self

    def adjust(self, test, alpha):
        predictions = self.predict(test)

        for (p, t) in zip(predictions, test):
            # What would I need to add to my prediction to be right?
            error = t.position - p["position"]
            
            # If there is a user or question we've never seen before, just return the global average
            if (t.userId not in self.userRatings) or (t.questionId not in self.questionRatings):
                self.averagePosition += alpha * error
                continue
    
            # If there is a category we've never seen before, report the global average
            if(t.questionCategory not in self.expectedPosition):
                self.averagePosition += alpha * error
                continue
            
            categorySlice = self.expectedPosition[t.questionCategory]
    
            # If a specific userRating hasn't been seen before, report the average for the given category
            userRating = self.userRatings[t.userId]
            if userRating not in categorySlice:
                self.averageCategoryPosition[t.questionCategory] += alpha * error
                continue
    
            # If a questionRating hasn't been seen before, go with the average position by userRating (and category)
            userSlice = categorySlice[userRating]
            questionRating = self.questionRatings[t.questionId]
            if questionRating not in userSlice:
                self.avergaeUserCategoryPosition[t.questionCategory][userRating] += alpha * error
                continue
    
            (sum, N) = userSlice[questionRating]
            
            if N != 0:
                userSlice[questionRating] = (((sum / N) + alpha * error) * N, N)

    def predict(self, test):
        predictions = []
        for t in test:
            # If there is a user or question we've never seen before, just return the global average
            if (t.userId not in self.userRatings) or (t.questionId not in self.questionRatings):
                predictions.append({ "id": t.id, "position": self.averagePosition})
                continue
    
            # If there is a category we've never seen before, report the global average
            if(t.questionCategory not in self.expectedPosition):
                predictions.append({ "id": t.id, "position": self.averagePosition})
                continue
            
            categorySlice = self.expectedPosition[t.questionCategory]
    
            # If a specific userRating hasn't been seen before, report the average for the given category
            userRating = self.userRatings[t.userId]
            if userRating not in categorySlice:
                predictions.append({ "id": t.id, "position": self.averageCategoryPosition[t.questionCategory]  })
                continue
    
            # If a questionRating hasn't been seen before, go with the average position by userRating (and category)
            userSlice = categorySlice[userRating]
            questionRating = self.questionRatings[t.questionId]
            if questionRating not in userSlice:
                predictions.append({ "id": t.id, "position": self.avergaeUserCategoryPosition[t.questionCategory][userRating]})
                continue
    
            (sum, N) = userSlice[questionRating]
            
            if N == 0:
                predictions.append({ "id": t.id, "position": 39.0})
            else:
                predictions.append({ "id": t.id, "position": sum / N})

        # After looking in Excel, try and "unbias" the results        
        for p in predictions:
            key = round(p["position"]/10.0)*10
            if key in self.adjustments:
                p["position"] += self.adjustments[key]
             
        for p in predictions:
            key = round(p["position"]/10.0)*10
            if key in self.adjustments2:
                p["position"] += self.adjustments2[key]
           
        for p in predictions:
            key = round(p["position"]/10)*10
            if key in self.adjustments3:
                p["position"] += self.adjustments3[key]
 
        for p in predictions:
            key = round(p["position"]/10)*10
            if key in self.adjustments4:
                p["position"] += self.adjustments4[key]
  


  
        for (t, p) in zip(test, predictions):
            if t.id not in self.questionRatings or self.questionRatings[t.id] != 1:
                continue
             
            if t.userId not in self.userRatings or self.userRatings[t.userId] != 2:
                continue
 
            key = round(p["position"]/10)*10
            if key == 50:
                if t.questionCategory == "Fine Arts":
                    p["position"] += -3.7
                if t.questionCategory == "History":
                    p["position"] += 2.6
                if t.questionCategory == "Mathematics":
                    p["position"] += 30
                if t.questionCategory == "Physics":
                    p["position"] += -21.8
                if t.questionCategory == "Social Studies":
                    p["position"] += 7.8

                    
        return predictions
    
    def getSlice(self, fromDict, name, defaultValue):
        if name not in fromDict:
            fromDict[name] = defaultValue
            
        return fromDict[name]
    
    def getAveragePosition(self, expectedPosition):
        (sum, N) = (0.0, 0.0)
        
        for (categoryName, categorySlice) in expectedPosition.items():
            for (userRating, userSlice) in categorySlice.items():
                for (questionRating, (s, n)) in userSlice.items():
                    sum += s
                    N += n
                    
        return sum / N
        
    def getAverageCategoryPosition(self, expectedPosition):
        D = {}
        for (categoryName, categorySlice) in expectedPosition.items():
            D[categoryName] = (0.0, 0.0)
            
            for (userRating, userSlice) in categorySlice.items():
                for (questionRating, (s, n)) in userSlice.items():
                    (S, N) = D[categoryName]
                    D[categoryName] = (S + s, N + n)
                    
            (S, N) = D[categoryName] 
            D[categoryName] = S / N
            
        return D
    
    def getAvgPositionByUserRatingAndCategory(self, expectedPosition):
        D = {}
        for (categoryName, categorySlice) in expectedPosition.items():
            D[categoryName] = {}
            
            for (userRating, userSlice) in categorySlice.items():
                D[categoryName][userRating] = (0.0, 0.0)
    
                for (questionRating, (s, n)) in userSlice.items():
                    (S, N) = D[categoryName][userRating]
                    D[categoryName][userRating] = (S + s, N + n)
                    
                (S, N) = D[categoryName][userRating]
                D[categoryName][userRating] = S / N
    
        return D

def getRefinedModel(dataset, train, userGran, quesGran):
    # Take would we would use for training and split into a fit training set and a remainder
    fitTrain, remainder = dataset.splitTrainTest(train, len(train)/8)
    
    # Take the remainder and split into adjustment training and residual remainder
    remainder, adjustTrain = dataset.splitTrainTest(remainder, len(remainder)/4)
    
    # Take the remainder and split into adjustment training and validation training
    adjustTest, validateTest = dataset.splitTrainTest(remainder, len(remainder)/2)

    model = RatingModel(userGran, quesGran)
    model.fit(dataset, fitTrain)
    
    adjustStandard = float("inf")
    validateStandard = float("inf")

    for x in adjustTrain:
        # Measure the MSE before making any changes on adjust and validation sets
        before = meanSquareError(model.predict(adjustTest), adjustTest)
        validateBefore = meanSquareError(model.predict(validateTest), validateTest)

        # Create a copy of the model, and try nudging a parameter by some amount
        offshot = deepcopy(model)
        offshot.adjust([x], 0.1)
        
        # Measure the MSE after for the adjust and validation sets
        after = meanSquareError(offshot.predict(adjustTest), adjustTest)
        validateAfter = meanSquareError(offshot.predict(validateTest), validateTest)

        # If both the adjust and validate MSE have improved, then keep the change 
        # with the belief we should see an improvement in the test set
        if( after < before and after < adjustStandard ):
            if( validateAfter < validateBefore and validateAfter < validateStandard):
                
                if after >= adjustStandard: raise ValueError
                if validateAfter >= validateStandard: raise ValueError
                
                print("%f %f" % (after, validateAfter))
                
                model = offshot
                adjustStandard = after
                validateStandard = validateAfter

    return model

def applyPredictedCorrections(model, fTrain, fTest, pTrain, pTest):
    errorX = []
    errorY = []
    for (f, p) in zip(fTrain, pTrain):
        actual = f.position
        predicted = p["position"]
        error = actual - predicted

        sample = {
                  "uRating": "NA" if f.userId not in model.userRatings else model.userRatings[f.userId],
                  "qRating": "NA" if f.questionId not in model.questionRatings else model.questionRatings[f.questionId],
                }

        errorX.append(sample)
        errorY.append(error)

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(errorX)

    errorModel = Lasso(alpha=.1, normalize=True)
    errorModel.fit(X, errorY)

    testX = []
    for (f, p) in zip(fTest, pTest):
        predicted = p["position"]

        sample = {
                  "uRating": "NA" if f.userId not in model.userRatings else model.userRatings[f.userId],
                  "qRating": "NA" if f.questionId not in model.questionRatings else model.questionRatings[f.questionId],
                }

        testX.append( sample )

    corrections = errorModel.predict(vectorizer.transform(testX))

    for (c, p) in zip(corrections, pTest):
        p["position"] += 0.0008 * c

    return pTest

def meanSquareError(predictions, test):
    return sqrt(average(map(lambda (a,b): pow(a["position"] - (b.position), 2), zip(predictions, test))))

def accuracy(predictions, test):
    return (len(predictions) + sum(map(lambda (a, b): sign(a["position"])*sign(b.position), zip(predictions, test)))) / float(2.0 * len(predictions))

def getLengthGranularity(train):
    positions = [abs(t.position) for t in train]
    lengths = [len(t.questionText) for t in train]

    corrOpt = float("-inf")
    jOpt = None

    for j in xrange(1, 50):
        D = {}
        for (p, l) in zip(positions, lengths):
            key = round(l/j, 0) * j
            if not key in D:
                D[key] = []

            D[key].append(p)

        pivot = { k : average(v) - j for (k,v) in D.items() }

        corr = pearsonr(pivot.keys(), pivot.values())
        if corr > corrOpt:
            corrOpt = corr 
            jOpt = j
    
    return jOpt

# def useQuestionLengthToPredictPosition(fTrain, fTest):
#     lengthGran = getLengthGranularity(fTrain)
# 
#     predictions = []
#     results = []
#     for t in fTest:
#         x = round(len(t.questionText) / lengthGran, 0) * lengthGran
#         predictedPos = 0.1311 * x + 1.7077
#  
#         ratingPredict = model.predict([t])[0]
#         predictions.append(ratingPredict)
#          
#         correctness = 1
#         if ratingPredict["position"] < 0:
#             correctness = -1
#  
#         results.append({"id": t.id, "position": predictedPos * correctness })
# 
#     print meanSquareError(predictions, fTest)
#     print accuracy(predictions, fTest)
# 
#     print meanSquareError(results, fTest)
#     print accuracy(results, fTest)

if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)    
    fTrain, fTest = dataset.splitTrainTest(train, len(train) / 5)

    model = RatingModel(7, 2)
    model.fit(dataset, fTrain)
    pTrain = model.predict(fTrain)
    pTest = model.predict(fTest)

    for i in xrange(0, 100):
        pTest = applyPredictedCorrections(model, fTrain, fTest, pTrain, pTest)
        
        print("%f %f" % (meanSquareError(pTest, fTest), accuracy(pTest, fTest)) )

#     A = set([t.questionId for t in fTrain])
#     B = set([t.questionId for t in fTest])
#     cap = A.intersection(B)
# 
#     trainLookup = {t.questionId : t for t in fTrain}
#     predictionLookup = { t.questionId : p for (t, p) in zip(fTest, predictions) }
# 
#     T = []
#     P = []
#     for questionId in cap:
#         T.append(trainLookup[questionId]) 
#         P.append(predictionLookup[questionId])
#     
#     print("overlap MSE: %f" % meanSquareError(P, T) )
#     print("overlap ACC: %f" % accuracy(P, T))
    

#     model = getRefinedModel(dataset, train, 7, 2)
#     predictions = model.predict(test)
#     fileFormat = GuessFormat()
#     fileFormat.serialize(predictions, "data/guess.csv") 