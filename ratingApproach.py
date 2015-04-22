from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from prediction import meanSquareError
from math import sqrt
from numpy.lib.function_base import average

def getSlice(fromDict, name, defaultValue):
    if name not in fromDict:
        fromDict[name] = defaultValue
        
    return fromDict[name]

def getSimplePredictions(dataset, train, test):
    byUser = dataset.groupByUser((train, test))
    userRatings = { userId : user.getRating()  for (userId, user) in byUser.items()  } 

    byQuestion = dataset.groupByQuestion((train, test))
    questionRatings = { questionId : question.getRating()  for (questionId, question) in byQuestion.items()  }

    # Train    
    expectedPosition = {}
    for t in train:
        categorySlice = getSlice(expectedPosition, t.questionCategory, {})
        
        userRating = userRatings[t.userId]
        userSlice = getSlice(categorySlice, userRating, {})

        questionRating = questionRatings[t.questionId]
        (sum, N) = getSlice(userSlice, questionRating, (0.0, 0.0))
            
        userSlice[questionRating] = (sum + t.position, N + 1.0)

    # Predict        
    predictions = []
    for t in test:
        if (t.userId not in userRatings) or (t.questionId not in questionRatings):
            predictions.append({ "id": t.id, "position": 39.0})
            continue

        if(t.questionCategory not in expectedPosition):
            predictions.append({ "id": t.id, "position": 39.0})
            continue
        
        categorySlice = expectedPosition[t.questionCategory]


        userRating = userRatings[t.userId]
        if userRating not in categorySlice:
            predictions.append({ "id": t.id, "position": 39.0})
            continue

        userSlice = categorySlice[userRating]
        questionRating = questionRatings[t.questionId]
        if questionRating not in userSlice:
            predictions.append({ "id": t.id, "position": 39.0})
            continue

        (sum, N) = userSlice[questionRating]
        
        if N == 0:
            predictions.append({ "id": t.id, "position": 39.0})
        else:
            predictions.append({ "id": t.id, "position": sum / N})
            
    return predictions

def meanSquareError(predictions, test):
    return sqrt(average(map(lambda (a,b): (a["position"] - b.position)*(a["position"] - b.position), zip(predictions, test))))


if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

    print("MSE: %f +- %f" % dataset.crossValidate(train, 5, lambda trainFold, testFold: 
          meanSquareError(getSimplePredictions(dataset, trainFold, testFold), testFold) ))

    predictions = getSimplePredictions(dataset, train, test)
    fileFormat = GuessFormat()
    fileFormat.serialize(predictions, "data/guess.csv") 