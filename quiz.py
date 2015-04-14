from csv import DictReader, DictWriter

import numpy as np
from numpy import array

import math

import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

import re

import argparse
import string
from collections import defaultdict
import operator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from data import Dataset
from fileFormats import TrainFormat, TestFormat, QuestionFormat, GuessFormat
from models import OriginalModel, User


#Simple edit distance computation
#Cite: http://rosettacode.org/wiki/Levenshtein_distance

def levenshteinDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]   

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)
    
    def getX(self, listData):
        # Input a list of TrainData or TestData
        # Output an array of vectors
                
        return self.vectorizer.fit_transform([ datum.getQuestionFragment() for datum in listData])
    
    def getSingleX(self, singleData):
        # Input a list of TrainData or TestData
        # Output a vector

        return self.vectorizer.transform([ singleData.getQuestionFragment() ])

    def getY(self, listData):
        # Input a list of TrainData
        # Output a vector of Y (isCorrect) values 
    
        return [(float(datum.position) > 0) for datum in listData]
    
    def getN(self, listData, boolA):
        # Input list of TrainData
        # Output array of positions N for only those questions that are isCorrect = boolA
        
        return [str(abs(float(datum.position))) for datum in listData if ((float(datum.position) > 0) == boolA)]
    
    def getSingleN(self, singleData, boolA):
        # Input list of TrainData
        # Output array of positions N for only those questions that are isCorrect = boolA
            
        return str(abs(float(singleData.position)))

def isCorrect(datum):
    # Input datum containing user answer and question answer 
    # Output true if correct, false otherwise
    # Edit distance is hard coded for now.
    
    # Case shouldn't factor into correctness
    userAnswer, questionAnswer = datum.userAnswer.lower(), datum.questionAnswer.lower()

    # An answer should still be equal if its off by punctuation: 
    #  e.g., mary's house == marys house
    #  e.g., key board == keyboard
    regex = re.compile("[^\w\s]|\s")
    userAnswer = re.sub(regex, "", userAnswer)
    questionAnswer = re.sub(regex, "", questionAnswer)
    #take away slight misspellings 
    editDistance = levenshteinDistance(userAnswer,questionAnswer)

    # User could be vauge, but correct, or overly specific, or implicitly, match identically
    return (userAnswer in questionAnswer) or (questionAnswer in userAnswer) or editDistance<3
        
def rms_train(userModels):
    # Input user models, users
    # Output root-mean-square score
     
    n = 0.0
    squareSum = 0.0
    for userId in userModels:
        model = userModels[userId]
        
        for x in model.user.train:
            actual = float(x.position)
            prediction = model.getExpectedPosition(x)
            squareSum += (prediction - actual) * (prediction - actual)
            n += 1.0

    return math.sqrt(squareSum/n)

def writeGuesses(userModels, test):
    # 2015-04-08 GEL This assumes the user independent model.. will need to 
    # adjust in future for different approach.
    model = userModels[0]
    
    guesses = []
    for testQuestion in test:
        print testQuestion.id
        print model.getExpectedPosition(testQuestion)
        guesses.append( { 'id': testQuestion.id, 'position': model.getExpectedPosition(testQuestion) } )

    guessFormat = GuessFormat()
    guessFormat.serialize(guesses, "guesses.csv")

def asSingleUser(train, test):
    user = User(0)
    user.test = test
    user.train = train
    return { 0: OriginalModel(user, Featurizer()) }

def asManyUsers(train, test):
    users = dataset.groupByUser((train, test))
    return { userId : OriginalModel(user, Featurizer()) for (userId, user) in users.items() }
# Have to incorporate user model
'''
def asUserCategory(train,test):
    users = dataset.groupByCategory((train, test))
    return { userId : OriginalModel(user, Featurizer()) for (userId, user) in users.items() }
'''

if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

    ## Uncomment this section for a user independent model
    userModels = asSingleUser(train, test)

    ## Uncomment for user dependent models
    #userModels = asManyUsers(train, test)
    
    ## Uncomment for category as user model
    #have to define
    #userModels = asUserCategory(train,test)

    writeGuesses(userModels, test)

#    print("Training set RMS score is %f" % rms_train(userModels))
