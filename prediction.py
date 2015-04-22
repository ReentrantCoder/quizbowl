from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from positionPrediction import getPositionPredictions, reportPositionPrediction
from correctnessPrediction import getCorrectnessPredictions, reportCorrectness
from numpy.lib.function_base import average
from math import sqrt

def meanSquareError(entities):
    return sqrt(average(map(lambda x : ( x["position"] - x["actual"] )*( x["position"] - x["actual"] ), entities)))

def getPredictions(dataset, training, test, alpha):
    positions = getPositionPredictions(dataset, training, test, alpha)
    correctness = getCorrectnessPredictions(dataset, training, test)

    return map(lambda (test, position, isCorrect): { 
        "id": test.id, 
        "position": position if isCorrect else -position, 
        "actual": test.position }, 
        zip(test, positions, correctness))
    
def reportOverall(dataset, training, alpha):
    print("Combined MSE: %f +- %f" % 
        dataset.crossValidate(training, 10, lambda trainingFold, testFold: 
            meanSquareError(getPredictions(dataset, trainingFold, testFold, alpha))))

if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    training, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    alpha = 1

    # Report expected scores
    reportCorrectness(dataset, training)
    reportPositionPrediction(dataset, training, alpha)
    reportOverall(dataset, training, alpha)

    # Evaluate the test set and write the results to disk.
    guessFormat = GuessFormat()
    guessFormat.serialize(getPredictions(dataset, training, test, alpha), "data/guess.csv")