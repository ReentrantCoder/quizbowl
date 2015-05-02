

from sklearn import cross_validation
from sklearn import datasets

import numpy

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
from sklearn.svm import SVR
from errorAnalysis import *
from sklearn import linear_model
import copy
from ratingApproach import *

#Very simple way to identify hints
def total_hints(text):
    
    count = 0
        
    words = text.split()
    for (prev, next) in zip(words, words[1:]):
        if prev.endswith(",") or prev.endswith(".") or prev.endswith(";") or prev.lower() in ["and", "or"]:
            count += 1

    return count

#Position Examples
def example_position(question,train=True):
    
    feat_dict = {'category': question.questionCategory,'question length': len(question.questionText.split()) ,'hints': total_hints(question.questionText),'answer':question.questionAnswer.lower()}
    
    if train:
        #target = question.position
        target = abs(question.position)
    else:
        target = 0 #unsurpervised, we don't know the info about our test set
    
    return feat_dict, target


def all_examples_position(questions,train=True): #from feature.py all examples
    
    examples = []
    targets = []
    for q in questions:
        
        
        (ex, tgt) = example_position(q,train)
        examples.append(ex)
        targets.append(tgt)
        
    return examples,targets

#Correctness Examples
def example_correct(question,train=True):
    
    feat_dict = {'category': question.questionCategory,'question length': len(question.questionText.split()) ,'answer':question.questionAnswer.lower(),'hints': total_hints(question.questionText),'position':abs(question.position)}
    
    
    
    if train:
        target = 1 if question.position>0 else -1
    else:
        target = 0
    
    return feat_dict, target


def all_examples_correct(questions,train=True):
    
    examples = []
    targets = []
    for q in questions:
        
        
        (ex, tgt) = example_correct(q,train)
        examples.append(ex)
        targets.append(tgt)
    
    return examples,targets

##Featurizers##

class Featurizer_position:
    def __init__(self):
        
        self.vectorizer = DictVectorizer()
    
    
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)
    
    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

class Featurizer_correct:
    def __init__(self):
        
        self.vectorizer = DictVectorizer()
    
    
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)
    
    def test_feature(self, examples):
        return self.vectorizer.transform(examples)



if __name__ == '__main__':
    
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    positions = [q.position for q in train]

    #For cross validation, change to test_size to 0.0 for learning on all the training data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, positions, test_size=0.4, random_state=None)


    questions = QuestionFormat()
    question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))
    

    #Uncomment this to run full test
    #X_test=test
    
    store = copy.deepcopy(X_test) #copy to update position before learning correctness
    store_train = copy.deepcopy(X_train)

    feat_pos = Featurizer_position()
    feat_cor = Featurizer_correct()

    features_pos,position = all_examples_position(X_train,True);
    features_cor,isCorrect = all_examples_correct(X_train,True);
    train_pos = feat_pos.train_feature(features_pos)
    train_cor = feat_cor.train_feature(features_cor)

    
    f_test,p_test = all_examples_position(X_test,False)
    t_test = feat_pos.test_feature(f_test)

    #Which model to use for regression to pick position
    #clf = linear_model.LinearRegression()
    #clf = linear_model.Ridge (alpha = 0.005)
    clf = linear_model.Lasso(alpha = 0.1)
    
    classifier_position = clf.fit(train_pos, position)
    y_lin = classifier_position.predict(t_test)
    y_lin_train = classifier_position.predict(train_pos)
    
    #Update the position in the test data once we have predicted it
    new_test = [x for x in X_test]
    for index,value in enumerate(y_lin):
        new_test[index].position = value
    
    #Which model to use for regression to pick correctness
    #Choose continuous over classifier
        #because of harsh penalty for incorrect correctness prediction
    #clf = linear_model.Lasso(alpha = 0.1)
    clf = linear_model.Ridge (alpha = 5)
    #clf = linear_model.LinearRegression()
    #logreg = linear_model.LogisticRegression(C=10000)
    #classifier_correct = logreg.fit(train_cor,isCorrect)

    f_test_cor,p_test_cor = all_examples_correct(new_test,False)
    t_test_cor = feat_cor.test_feature(f_test_cor)



    classifier_correct = clf.fit(train_cor,isCorrect)
    y_cor = classifier_correct.predict(t_test_cor)

    predictions_lin = []
    for index,value in enumerate(y_lin):
        predictions_lin.append({'id':X_test[index].id,'position': y_cor[index]*abs(value)})
    compute_error_analysis(predictions_lin,store,25)


    #Training Data
    print "Training Data"
    new_train = [x for x in X_train]
    for index,valye in enumerate(y_lin_train):
        new_train[index].position = value

    f_train_cor,p_train_cor = all_examples_correct(new_train,False)
    t_train_cor = feat_cor.test_feature(f_train_cor)

    y_train_cor = classifier_correct.predict(t_train_cor)
    
    predictions_lin_train = []
    for index,value in enumerate(y_lin_train):
        predictions_lin_train.append({'id':X_train[index].id,'position': y_train_cor[index]*abs(value)})
    compute_error_analysis(predictions_lin_train,store_train,25)


#Output the guesses
    fileFormat = GuessFormat()
    fileFormat.serialize(predictions_lin, "data/guess222.csv")




