from csv import DictReader, DictWriter

import numpy as np
from numpy import array

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

import math
from numpy.ma.core import mean
from numpy import sign

def pars(x):
    q = "question" + str(x['question'])
    u = "user" + str(x['user'])
    qu = [q, u]
    return (y for y in qu)

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10]))) 
            
def crossValidate(train, K = 5):
    bucket = len(train)/K

    values = []
    for k in xrange(0, K):
        random.shuffle(train)
        asTest = train[:bucket]
        asTrain = train[bucket:]      
        values.append( crossLearn(asTrain, asTest) )
    
    return mean(values) 
    
def crossLearn(train, test):

    feat = Featurizer()
    labels = ['39.298062750052644', '-39.298062750052644']    

    x_train = feat.train_feature(x['question'] for x in train)
    x_test = feat.test_feature(x['question'] for x in test)

    y_train = array(list(labels[0] if float(x['position']) > 0 else labels[1] for x in train))
    y_test  = array(list(x['position'] for x in test))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    return rms(lr, x_test, y_test)
    
def rms(classifier, data, actual):
    # Input classifier and training data that was reserved for testing
    # Output root-mean-square score
     
    n, squareSum = 0.0, 0.0
       
    predictions = classifier.predict(data) 
    for (x, y) in zip(predictions, actual):
        x, y = 20*sign(float(x)), float(y)
        squareSum += (x-y)*(x-y)
        n += 1.0

    return math.sqrt(squareSum/n)

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/train.csv", 'r')))
    test = list(DictReader(open("data/test.csv", 'r')))

    feat = Featurizer()
    labels = ['39.298062750052644', '-39.298062750052644']  

    x_train = feat.train_feature(x['question'] for x in train)
    x_test = feat.test_feature(x['question'] for x in test)

    y_train = array(list(labels[0] if float(x['position']) > 0 else labels[1] for x in train))
    #y_test  = array(list(labels.index(x['cat']) for x in test))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    #feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'position': pp}
        o.writerow(d)
        
    print crossValidate(train)
