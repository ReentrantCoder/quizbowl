from csv import DictReader, DictWriter

import numpy as np
from numpy import array
from fileFormats import *


import random

from random import shuffle

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

catposwrong =  {'Astronomy': -78.77142857,
            'Biology':	-77.93413174,
            'Chemistry': -75.8245614,
            'Earth Science':-77.57142857,
            'Fine Arts':	-92.69809524,
            'History':	-87.40901639,
            'Literature':	-95.17899638,
            'Mathematics':	-90.54545455,
            'Other':	-97.32,
            'Physics': -80.84519573,
            'Social Studies':	-88.05496183 }
            
catposright = { 'Astronomy':	77.33333333,
                'Biology':	75.45972222,
                'Chemistry':	75.59920107,
                'Earth Science':	77.46153846,
                'Fine Arts':	87.6202765,
                'History':	85.64456869,
                'Literature':	91.51880966,
                'Mathematics':	75.94285714,
                'Other':	89.55752212,
                'Physics':	81.92889289,
                'Social Studies':	84.0219363 }

catpos = { '-1.0': catposwrong, '1.0': catposright }



#analyzer that determines which features to use
class Analyzer:
    def __init__(self,q_id,u_id,category,keywords):
        self.q_id = q_id
        self.u_id = u_id
        self.category = category
        self.keywords = keywords
    
    def __call__(self, feature_list):
        if self.q_id:
            yield feature_list.pop()
        if self.u_id:
            yield feature_list.pop()
        if self.category:
            yield feature_list.pop()
        if self.keywords:
            words = feature_list.pop()
            for w in words:
                yield w

#Modified from features.py.  All the possible features of an question
def example(answer_info, question_info,train=True):
    
    q = "question"+str(answer_info['question']) #question id -> keywords, category
    u = "user" + str(answer_info['user'])
    
    c = "category" + str(question_info[answer_info['question']]['category'])
    #Need to add keywords
    '''
    k = []
    for (key,value) in question_info[answer_info['question']]['blob']:
        k.append(value)
    '''
    qu = [q,u,c]

    #This may be unnecessary if we are always using the average answers instead of the one provided
    if train:
        target = str(sign(float(answer_info['position'])))
    else:
        target = 0 #unsurpervised, we don't know the info about our test set
    
    return qu, target


def all_examples(examples,limit,question_info,train=True): #from feature.py all examples
    ex_num = 0
    for ii in examples:
        ex_num += 1
        if limit > 0 and ex_num > limit:
            break
        
        ex, tgt = example(ii,question_info,train)
        
        yield ex, tgt


class Featurizer:
    def __init__(self):
        analyzer = Analyzer(True,True,True,False)
        self.vectorizer = CountVectorizer(analyzer=analyzer,binary=True)
    

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

def accuracy(classifier, x, y, examples):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("\t".join(kTAGSET[1:]))
    for ii in cm:
        print("\t".join(str(x) for x in ii))

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
    labels = ['80', '-80']
    
    x_train = feat.train_feature(ex for ex, tgt in
                                 all_examples(train,len(train),question_info,True)) #all the words
    
    
    x_test = feat.test_feature(ex for ex, tgt in
                               all_examples(test,len(train),question_info,False))
                               
    examp = [ex for ex, tgt in
                               all_examples(test,len(train),question_info,False)]
                               
    y_train = array(list(tgt for ex, tgt in all_examples(train[0:flags.limit],flags.limit,question_info,True)))

    y_test  = array(list(x['position'] for x in train))


    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    return rms(lr, x_test, examp, y_test)

def rms(classifier, data, ex, actual):
    # Input classifier and training data that was reserved for testing
    # Output root-mean-square score
    
    n, squareSum = 0.0, 0.0
    
    predictions = classifier.predict(data)
    for (x, d, y) in zip(predictions, ex, actual):
        c = d[2].replace('category', '')
        z, y = catpos[x][c], float(y)
        squareSum += (z-y)*(z-y)
        n += 1.0
    
    return math.sqrt(squareSum/n)

if __name__ == "__main__":
    
    #Added for a little debugging
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--limit', default=-1, type=int,
                        help="How many questions to use")
    parser.add_argument('--computeTotal', default=True,
                        help="Compute Full Output")
                        
                    
    flags = parser.parse_args()
    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/train.csv", 'r')))
    test = list(DictReader(open("data/test.csv", 'r')))
    totalTrain = len(train)
    shuffle(train) #randomize test data so training and validation have similar distributions
    
    questions = QuestionFormat()
    question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))
   
    if flags.computeTotal:
       flags.limit = totalTrain #testing on everything
    

    feat = Featurizer()
    #labels = ['39.298062750052644', '-39.298062750052644']  
    
    x_train = feat.train_feature(ex for ex, tgt in
                                 all_examples(train[0:flags.limit],flags.limit,question_info,True)) #all the words
        
        
    x_test = feat.test_feature(ex for ex, tgt in
                               all_examples(test,flags.limit,question_info,False))
                               
    examp = [ex for ex, tgt in all_examples(test,len(train),question_info,False)]
                               
    #y_train = array(list(labels[0] if float(x['position']) > 0 else labels[1] for x in train))
                                 
    y_train = array(list(tgt for ex, tgt in all_examples(train[0:flags.limit],flags.limit,question_info,True)))
    
    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    #feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
    o.writeheader()
    for ii, pp, ee in zip([x['id'] for x in test], predictions, examp):
        c = ee[2].replace('category', '')
        pp = catpos[pp][c]
        d = {'id': ii, 'position': pp}
        o.writerow(d)
            
    print crossValidate(train)
    

