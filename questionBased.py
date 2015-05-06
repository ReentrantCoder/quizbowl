

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
        if prev.endswith(",") or prev.endswith(".") or prev.endswith(";"):
            #or prev.lower() in ["and", "or"]:
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
    
    #text = question.questionText.split()
    #print len(text)
    #lead_pos = int(abs(question.position))
    #print lead_pos
    #leading_word = text[lead_pos-1]
   
   #trailing_word = text[lead_pos-2]

    
    feat_dict = {'category': question.questionCategory,'question length': len(question.questionText.split()) ,'answer':question.questionAnswer.lower(),'hints': total_hints(question.questionText),'position':abs(question.position)}#,'word':leading_word}#,'leading2':trailing_word}
    #'answer':question.questionAnswer.lower(),
    
    
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

    def get_features(self):
        return self.vectorizer.get_feature_names()

class Featurizer_correct:
    def __init__(self):
        
        self.vectorizer = DictVectorizer()
    
    
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)
    
    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def get_features(self):
        return self.vectorizer.get_feature_names()


def process_known_users(train):
    #map user id to tuple (question_answered, right,wrong)
    user_map = {t.userId:[0,0,0] for t in train}
    for t in train:
        user_map[t.userId][0]+=1
        if t.position < 0:
            user_map[t.userId][2]+=1
        else:
            user_map[t.userId][1]+=1

    return user_map

#def process_known_questions(train):

def get_known_users(train,cap):
    users = set([])
    for t in train:
        if t.userId in cap:
            users = users | set([t])
    return users

def process_known_questions(train):
    #map question id to tuple (question_answered, right,wrong)
    question_map = {t.questionId:[0,0,0] for t in train}
    for t in train:
        question_map[t.questionId][0]+=1
        if t.position < 0:
            question_map[t.questionId][2]+=1
        else:
            question_map[t.questionId][1]+=1

    return question_map

def get_known_questions(train,cap):
    questions = set([])
    for t in train:
        if t.questionId in cap:
            questions = questions | set([t])
    return questions



if __name__ == '__main__':
    
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)
    actual_test, dumb_dumb = dataset.getTrainingTest("data/TestResult.csv", "data/TestResult.csv", "data/questions.csv", -1)
    positions = [q.position for q in train]

    #For cross validation, change to test_size to 0.0 for learning on all the training data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, positions, test_size=0.0, random_state=None)
    



    questions = QuestionFormat()
    question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))
    

    #Uncomment this to run full test
    X_test=test
    
    # Figure out what percent of users are in both train and test sets
    '''
    A = set([ t.userId for t in X_train ])
    B = set([ t.userId for t in X_test ])
    cap = A.intersection(B)
    
    known_users = process_known_users(get_known_users(X_train,cap))
    
    C = set([ t.questionId for t in X_train ])
    D = set([ t.questionId for t in X_test ])
    cap_questions = C.intersection(D)
    
    known_questions = process_known_questions(get_known_questions(X_train,cap_questions))
    '''
    #print known_users
    #exit(-1)
                       
    store = copy.deepcopy(X_test) #copy to update position before learning correctness
    store_train = copy.deepcopy(X_train)

    feat_pos = Featurizer_position()
    feat_cor = Featurizer_correct()

    features_pos,position = all_examples_position(X_train,True);
    features_cor,isCorrect = all_examples_correct(X_train,True);
    #print "IS CORRECT"
    #print isCorrect
    train_pos = feat_pos.train_feature(features_pos)
    train_cor = feat_cor.train_feature(features_cor)
    features_position = feat_pos.get_features()
    features_correct = feat_cor.get_features()

    
    f_test,p_test = all_examples_position(X_test,False)
    t_test = feat_pos.test_feature(f_test)

    #Which model to use for regression to pick position
    #clf = linear_model.LinearRegression()
    #clf = linear_model.Ridge (alpha = 0.005)
    clf = linear_model.Lasso(alpha = 0.1)
    
    classifier_position = clf.fit(train_pos, position)
    
    matter_position = []
    for index,c in enumerate(classifier_position.coef_):
        if c > 0.0001 or c<-0.0001:
            matter_position.append([index,c])



    #print classifier_position.get_params()
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
    matter_correct = []


    for index,c in enumerate(classifier_correct.coef_):
        if c > 0.0001 or c<-0.0001:
            matter_correct.append([index,c])

    y_cor = classifier_correct.predict(t_test_cor)

    predictions_lin = []
    abs_position = []
    correct_conf = []
    total_change = 0
    min = 10
    max = 0
    for index,value in enumerate(y_lin):
        '''
        user_id = X_test[index].userId
        question_id = X_test[index].questionId
        total_change_old = total_change
        
        if known_users.has_key(user_id):
            info = known_users[user_id]
            if info[2]/info[0] > 0.3 and y_cor[index]>0:
                y_cor[index] = -1#*y_cor[index]
                total_change+=1
            elif info[1]/info[0] > 0.5 and y_cor[index]<0:
                y_cor[index] = 1#-1*-y_cor[index]
                total_change+=1
        
        
        if known_questions.has_key(question_id) and total_change_old==total_change:
            info = known_questions[question_id]
            if info[2]/info[0] > 0.5 and y_cor[index]>0:
                y_cor[index] = -1*-y_cor[index]
                total_change+=1
            
            elif info[1]/info[0] > 0.5 and y_cor[index]<0:
                y_cor[index] = -1*-y_cor[index]
                total_change+=1
        
        '''

        '''
        v = abs(value)/len(X_test[index].questionText.split())
        if v>max:
            max=v
        if v<min:
            min =v
        if abs(value)/len(X_test[index].questionText.split()) < 0.72 and y_cor[index]>0:
            y_cor[index] = -y_cor[index]
            total_change+=1
        '''
        predictions_lin.append({'id':X_test[index].id,'position': y_cor[index]*value})
        abs_position.append({'id':X_test[index].id,'position': value})
        correct_conf.append({'id':X_test[index].id,'position': y_cor[index]})
        #predictions_lin.append({'id':X_test[index].id,'position': value})
    compute_error_analysis(predictions_lin,actual_test,25)
    #compute_error_analysis(abs_position,actual_test,25)

    '''
    print "total change"
    print total_change
    print max
    print min
    '''

    #Training Data
    print "Training Data"
    new_train = [x for x in X_train]
    for index,value in enumerate(y_lin_train):
        '''
        if abs(value)>len(X_train[index].questionText.split()):
            value = len(X_train[index].questionText.split())/2
        '''
        new_train[index].position = value

    f_train_cor,p_train_cor = all_examples_correct(new_train,False)
    t_train_cor = feat_cor.test_feature(f_train_cor)

    y_train_cor = classifier_correct.predict(t_train_cor)

    total_change_t = 0
    min_t = 10
    max_t = 0
    predictions_lin_train = []
    for index,value in enumerate(y_lin_train):
        '''
        user_id = X_train[index].userId
        question_id = X_train[index].questionId
        total_change_t_old = total_change_t
        
        if known_users.has_key(user_id):
            info = known_users[user_id]
            if info[2]/info[0] > 0.3 and y_train_cor[index]>0:
                y_train_cor[index] = -1#*y_train_cor[index]
                total_change_t+=1
            elif info[1]/info[0] > 0.5 and y_train_cor[index]<0:
                y_train_cor[index] = 1#-1*-y_train_cor[index]
                total_change_t+=1


        if known_questions.has_key(question_id) and total_change_t_old==total_change_t:
            info = known_questions[question_id]
            if info[2]/info[0] > 0.5 and y_train_cor[index]>0:
                y_train_cor[index] = -1*y_train_cor[index]
                total_change_t+=1
            
            elif info[1]/info[0] > 0.5 and y_train_cor[index]<0:
                y_train_cor[index] = -1*y_train_cor[index]
                total_change_t+=1
        '''
        

        '''
        v = abs(value)/len(X_train[index].questionText.split())
        if v>max_t:
            max_t=v
        if v<min_t:
            min_t =v
        if abs(value)/len(X_train[index].questionText.split()) > 0.72 and y_train_cor[index]>0:
            y_train_cor[index] = -y_train_cor[index]
        '''
        predictions_lin_train.append({'id':X_train[index].id,'position': y_train_cor[index]*abs(value)})
        
        #compute_error_analysis(predictions_lin_train,store_train,25)


#Feature Data
    print "Position"
    for i in matter_position:
        print i[1],features_position[i[0]]
   
    print "Correct"
    print "effective features correct"
    print len(matter_correct)
    matter_correct1 = []
    for c in matter_correct:
        matter_correct1.append((c[1],features_correct[c[0]]))
    
    mc = sorted(matter_correct1, key=lambda tup: tup[0])
    for m in mc:
        #if m[1][0:2] != 'an':
        print m


    '''
    print "total change"
    print total_change_t
    print max_t
    print min_t
    '''


#Output the guesses
    fileFormat = GuessFormat()
#fileFormat.serialize(predictions_lin, "data/guessFinalNo12.csv")
#fileFormat.serialize(abs_position, "data/guessPosition.csv")
    fileFormat.serialize(correct_conf, "data/guessCorrectness.csv")






