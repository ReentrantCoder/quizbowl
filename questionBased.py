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



def load_data():

    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)

    questions = QuestionFormat()
    question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))

    return train,test,question_info


def total_hints(text):
    
    count = 0
        
    words = text.split()
    for (prev, next) in zip(words, words[1:]):
        if prev.endswith(",") or prev.endswith(".") or prev.endswith(";") or prev.lower() in ["and", "or"]:
            count += 1

    return count

#Modified from features.py.  All the possible features of an question
def example(question,train=True):
    
    feat_dict = {'category': question.questionCategory,'question length': len(question.questionText.split()) ,'hints': total_hints(question.questionText)}
    
    if train:
        target = question.position
    else:
        target = 0 #unsurpervised, we don't know the info about our test set
    
    return feat_dict, target


def all_examples(questions,train=True): #from feature.py all examples
    
    examples = []
    targets = []
    for q in questions:
        
        
        (ex, tgt) = example(q,train)
        examples.append(ex)
        targets.append(tgt)
        
    return examples,targets

class Featurizer:
    def __init__(self):
        
        self.vectorizer = DictVectorizer()
    
    
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)
    
    def test_feature(self, examples):
        return self.vectorizer.transform(examples)


def create_features():
    train,test,question_info = load_data()

if __name__ == '__main__':
    
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    positions = [q.position for q in train]
    print len(positions)
    print len(train)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, positions, test_size=0.4, random_state=0)

    
    questions = QuestionFormat()
    question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))
    
    


    feat = Featurizer()

    features,position = all_examples(X_train,True);
    t = feat.train_feature(features)
    #feat.test_feature(y_train)
    print len(X_train)
    print len(features)
    print len(y_train)
    #print position
    
    f_test,p_test = all_examples(X_test,False)
    t_test = feat.test_feature(f_test)
    #print features
    #print X_train
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(t, y_train).predict(t_test)
    predictions = []
    for index,value in enumerate(y_rbf):
        predictions.append({'id':X_test[index].id,'position': value})
    compute_error_analysis(predictions,X_test,25)
    #y_lin = svr_lin.fit(features, positions).predict(X_test)
    #y_poly = svr_poly.fit(features, positions).predict(X_test)




    




