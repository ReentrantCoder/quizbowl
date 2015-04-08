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

# kTAGSET = ["", "Ast", "Bio", "Che", "Ear", "Fin", "Geo", "His", "Lit", "Mat", "Oth", "Phy", "Sci", "SSc", "SSt" ]
#                 
# def pars(examples):
#     selpmaxe = []
#     for text in examples:
#         txet = []
#         blah = text.split()
#         for word in text.split():
#         #for i in range(len(blah)):
#             txet.append(word)
#             '''
#             for j in range(len(blah)):
#                 if j > i:
#                     txet.append(blah[i]+blah[j])
#                 '''
#             if len(txet) > 1:
#               txet.append(prev+word)
#             prev = word  
#         selpmaxe.append(" ".join(txet))
#     return (x for x in selpmaxe)
# 
# class Featurizer:
#     def __init__(self):
#         self.vectorizer = CountVectorizer(binary=True)
# 
#     def train_feature(self, examples):
#         ##TODO: Make an Analyzer.
#         #This code only affects train_features. 
#         #Needs to affect test_features as well.
#         #Add features (don't remove them) to improve
#         return self.vectorizer.fit_transform(pars(examples))
# 
#     def test_feature(self, examples):
#         return self.vectorizer.transform(pars(examples))
# 
#     def show_top10(self, classifier, categories):
#         feature_names = np.asarray(self.vectorizer.get_feature_names())
#         for i, category in enumerate(categories):
#             top10 = np.argsort(classifier.coef_[i])[-10:]
#             print("%s: %s" % (category, " ".join(feature_names[top10])))
# 
# def accuracy(classifier, x, y, examples):
#     predictions = classifier.predict(x)
#     cm = confusion_matrix(y, predictions)
# 
#     print("Accuracy: %f" % accuracy_score(y, predictions))
# 
#     print("\t".join(kTAGSET[1:]))
#     for ii in cm:
#         print("\t".join(str(x) for x in ii))   



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
        # Input a list of TrainData or TestData
        # Output a vector of Y (isCorrect) values 
    
        return [isCorrect(datum) for datum in listData]
    
    def getN(self, listData, boolA):
        # Input list of TrainData
        # Output array of positions N for only those questions that are isCorrect = boolA
        
        return [datum.position for datum in listData if isCorrect(datum) == boolA]
    
    def getSingleN(self, singleData, boolA):
        # Input list of TrainData
        # Output array of positions N for only those questions that are isCorrect = boolA
            
        return singleData.position

def isCorrect(datum):
    # Input datum containing user answer and question answer 
    # Output true if correct, false otherwise

    # Case shouldn't factor into correctness
    userAnswer, questionAnswer = datum.userAnswer.lower(), datum.questionAnswer.lower()

    # An answer should still be equal if its off by punctuation: 
    #  e.g., mary's house == marys house
    #  e.g., key board == keyboard
    regex = re.compile("[^\w\s]|\s")
    userAnswer = re.sub(regex, "", userAnswer)
    questionAnswer = re.sub(regex, "", questionAnswer)

    # User could be vauge, but correct, or overly specific, or implicitly, match identically
    return (userAnswer in questionAnswer) or (questionAnswer in userAnswer)
        
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

def writeGuesses(userModels):
    guesses = []
    for userId in userModels:
        model = userModels[userId]
        for x in model.user.test:
            guesses.append( { 'id': x.questionId, 'position': model.getExpectedPosition(x) } )
     
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

if __name__ == "__main__":
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", 500)
    
    ## Uncomment this section for a user independent model
    userModels = asSingleUser(train, test)

    ## Uncomment for user dependent models
    #userModels = asManyUsers(train, test)

    print("Training set RMS score is %f" % rms_train(userModels))

#     # Cast to list to keep it all in memory
#     train = list(DictReader(open("train.csv", 'r')))
#     test = list(DictReader(open("test.csv", 'r')))

#     feat = Featurizer()
# 
#     #This discovers all the categories (Biology, Math, etc.)
#     labels = [] 
#     for line in train:
#         if not line['cat'] in labels:
#             labels.append(line['cat'])  
# 
#     random.shuffle(train)
#     #test = train[len(train) - 4999:]
#     #train = train[:len(train) - 5000]
#     x_train = feat.train_feature(x['text'] for x in train)
#     x_test = feat.test_feature(x['text'] for x in test)
# 
#     y_train = array(list(labels.index(x['cat']) for x in train))
#     #y_test  = array(list(labels.index(x['cat']) for x in test))
# 
#     # Train classifier
#     lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#     lr.fit(x_train, y_train)
#     
#     print x_train.shape
#     print y_train.shape
#     print y_train
# 
#     feat.show_top10(lr, labels)
#     '''
#     predictions = lr.predict(x_test)
#     o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
#     o.writeheader()
#     for ii, pp in zip([x['id'] for x in test], predictions):
#         d = {'id': ii, 'cat': labels[pp]}
#         o.writerow(d)
#    
#     print("TRAIN\n-------------------------")
#     accuracy(lr, x_train, y_train, (x['text'] for x in train))
#     print("TEST\n-------------------------")
#     accuracy(lr, x_test, y_test, (x['text'] for x in test))
#     '''
