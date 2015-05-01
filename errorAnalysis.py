#Take in predicted data and actual data
#Take in questions
#Take in difference we care about (how much is too much to be off by

from __future__ import division

from numpy import array
from fileFormats import *


import random

import re

import argparse
import string
from collections import defaultdict
import operator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import math
from numpy.ma.core import mean
from numpy import sign,sqrt,mean,power,average

import fileFormats
import data

questions = QuestionFormat()
question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))

#Assume actual in TrainingData Format
#Assume predicted is a dictionary of the form {"id": id, "position": predicted position}
#Returns Ordered List of question,error_posstion pairs
#error_position: how far off we are in terms of absolute value of predicted position

def compute_error_analysis(predicted,actual,maxDiff):
    missedQuestions = position_off(predicted,actual,maxDiff)
    wrong_answer = incorrect_predicted_answer(predicted,actual)
    cat_off = categories(missedQuestions)
    cat_answer = categories(wrong_answer)
    all_cat = all_categories(actual)
    print "total questions"
    print len(predicted)
    print len(missedQuestions)
    print len(wrong_answer)
    print cat_off
    print cat_answer
    print all_cat
    print percent_category_missed(cat_off,all_cat,)
    print percent_category_missed(cat_answer,all_cat,)
    print mse(predicted,actual)

def mse(predicted,actual):
    
    return sqrt(average(map(lambda (a,b): pow(a["position"] - (b.position), 2), zip(predicted, actual))))

def position_off(predicted,actual,maxDiff):
    missedQuestions = []
    for index,value in enumerate(predicted):
        
        #print (value['id'], abs(value['position']))
        #print (actual[index].id,abs(actual[index].position))
        #print
        diff = abs(abs(value['position'])-abs(actual[index].position))
        if diff > maxDiff:
            missedQuestions.append((question_info[actual[index].questionId],diff))
    print "total questions"
    print len(predicted)
    print(len(missedQuestions))
    ordered = sorted(missedQuestions, key=lambda pos: pos[1])
    #print ordered[0]
    return ordered

def incorrect_predicted_answer(predicted,actual):
    wrong_answer = []
    for index,value in enumerate(predicted):
        
        #print sign(value['position'])
        #print sign(actual[index].position)
        #print sign(value['position'])!=sign(actual[index].position)
        if sign(value['position'])!=sign(actual[index].position):
            wrong_answer.append((question_info[actual[index].questionId],(value['position'],actual[index].position)))
    return wrong_answer

def all_categories(actual):
    cat = {}
    for a in actual:
        key = question_info[a.questionId]['category']
        #print key
        if cat.has_key(key):
            cat[key]+=1
        else:
            cat[key]=1
    return cat

def categories(wrong):
    cat = {}
    for w in wrong:
        if cat.has_key(w[0]['category']):
            cat[w[0]['category']]+=1
        else:
            cat[w[0]['category']]=1

    return cat

def percent_category_missed(wrong,total):
    percent_missed = {}
    for key,value in total.iteritems():
        if wrong.has_key(key):
            percent_missed[key] = (wrong[key]/value)*100
        else:
            percent_missed[key] = 0
    return percent_missed

#def length(wrong):

    


