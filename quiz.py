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

kTAGSET = ["", "Ast", "Bio", "Che", "Ear", "Fin", "Geo", "His", "Lit", "Mat", "Oth", "Phy", "Sci", "SSc", "SSt" ]
                
def pars(examples):
    selpmaxe = []
    for text in examples:
        txet = []
        blah = text.split()
        for word in text.split():
        #for i in range(len(blah)):
            txet.append(word)
            '''
            for j in range(len(blah)):
                if j > i:
                    txet.append(blah[i]+blah[j])
                '''
            if len(txet) > 1:
              txet.append(prev+word)
            prev = word  
        selpmaxe.append(" ".join(txet))
    return (x for x in selpmaxe)

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def train_feature(self, examples):
        ##TODO: Make an Analyzer.
        #This code only affects train_features. 
        #Needs to affect test_features as well.
        #Add features (don't remove them) to improve
        return self.vectorizer.fit_transform(pars(examples))

    def test_feature(self, examples):
        return self.vectorizer.transform(pars(examples))

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

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    #This discovers all the categories (Biology, Math, etc.)
    labels = [] 
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])  

    random.shuffle(train)
    #test = train[len(train) - 4999:]
    #train = train[:len(train) - 5000]
    x_train = feat.train_feature(x['text'] for x in train)
    x_test = feat.test_feature(x['text'] for x in test)

    y_train = array(list(labels.index(x['cat']) for x in train))
    #y_test  = array(list(labels.index(x['cat']) for x in test))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    
    print x_train.shape
    print y_train.shape
    print y_train

    feat.show_top10(lr, labels)
    '''
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
   
    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train, (x['text'] for x in train))
    print("TEST\n-------------------------")
    accuracy(lr, x_test, y_test, (x['text'] for x in test))
    '''
