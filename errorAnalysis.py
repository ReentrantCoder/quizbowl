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
from questionBased import *

questions = QuestionFormat()
question_info = questions.generatorToDict((questions.deserialize("data/questions.csv")))

#Assume actual in TrainingData Format
#Assume predicted is a dictionary of the form {"id": id, "position": predicted position}

def print_dictionary(dic):
    for (k,v) in dic.iteritems():
        print str(k) + " -> " + str(v)



def compute_error_analysis(predicted,actual,maxDiff):
    missedQuestions = position_off(predicted,actual,maxDiff)
    wrong_answer = incorrect_predicted_answer(predicted,actual)
    cat_off = categories(missedQuestions)
    cat_answer = categories(wrong_answer)
    all_cat = all_categories(actual)
    r_w = right_wrong(predicted,actual)
    w_r = wrong_right(predicted,actual)
    #print "total questions"
    #print len(predicted)
    #print "missed questions"
    #print len(missedQuestions)
    #print "wrong answer"
    #print len(wrong_answer)
    print "accuracy"
    print (1-(len(wrong_answer)/len(predicted)))*100
    
    print "All categories"
    print_dictionary(all_cat)

    #print "Position off"
    #print print_dictionary(percent_category_missed(cat_off,all_cat))
    #print "Answer off"
    #print_dictionary(percent_category_missed(cat_answer,all_cat))
    
    print "mean squared error"
    mse_,max_bad = mse(predicted,actual)
    print mse_
    for m in max_bad[-5:]:
        print m[1][0].questionCategory
        print m[1][0].questionText
        print m[1][0].position
        print "length: " + str(len(m[1][0].questionText.split()))
        print m[1][0].id
        print "hints: " + str(total_hints(m[1][0].questionText))
        print m[1][1]['position']
    
    #exit(-1)
    #print max_bad[1][0].questionCategory
    #print max_bad[1][1]

    print "predicted right actually wrong"
    print (r_w/len(predicted))*100

    print "predicted wrong actually right"
    print (w_r/len(predicted))*100

    print "questions right/wrong"
    my_dict = categories_right_wrong(actual,all_cat)
    print_dictionary(my_dict)

    #dict_to_csv("test.csv",my_dict)
    dict_1 = percent_category_missed(cat_answer,all_cat)

    #dict_to_csv("answeroff.csv",dict_1)

    dict_to_csv("positionoff.csv",categories_position_off(predicted,actual,all_cat))

    print "Position off"
    for m in missedQuestions[-5:]:
        #print m
        print m[0]['category']
        print m[0]['question']
        print "length: " + str(len(m[0]['question'].split()))
        #print m[0].id
        print "hints: " + str(total_hints(m[0]['question']))
        
        print m[1]
        print m[2]
        print m[3]


def dict_to_csv(fileName,my_dict):
    with open(fileName, 'wb') as outfile:  # Just use 'w' mode in 3.x
        writer = csv.writer(outfile)
        writer.writerow(my_dict.keys())
        writer.writerows(zip(*my_dict.values()))


def right_wrong(predicted,actual):
    i=0
    for index,value in enumerate(predicted):
        if value['position']>0 and actual[index].position<0:
            i+=1
    return i

def wrong_right(predicted,actual):
    i=0
    for index,value in enumerate(predicted):
        if value['position']<0 and actual[index].position>0:
            i+=1
    return i

def mse(predicted,actual):
    squares = []
    max_bad = []
    for index,value in enumerate(actual):
        how_bad = pow(predicted[index]["position"] - (value.position), 2)
        squares.append(how_bad)
        max_bad.append((how_bad,(value,predicted[index])))
    
    

    return sqrt(average(squares)),sorted(max_bad, key=lambda tup: tup[0])
    
#return sqrt(average(map(lambda (a,b): pow(a["position"] - (b.position), 2), zip(predicted, actual))))

#Returns Ordered List of question,error_posstion pairs
#error_position: how far off we are in terms of absolute value of predicted position
def position_off(predicted,actual,maxDiff):
    missedQuestions = []
    for index,value in enumerate(predicted):
        
        #print (value['id'], abs(value['position']))
        #print (actual[index].id,abs(actual[index].position))
        #print
        diff = abs(abs(value['position'])-abs(actual[index].position))
        if diff > maxDiff:
            missedQuestions.append((question_info[actual[index].questionId],
                                    (value['position'],actual[index].position,diff),
                                    actual[index].questionId,actual[index].id))
#print "total questions"
#    print len(predicted)
#    print(len(missedQuestions))
    ordered = sorted(missedQuestions, key=lambda pos: pos[1][2])
    #print ordered[0]
    return ordered

#Returns List of questions that whose prediction and answer do not match correctness
def incorrect_predicted_answer(predicted,actual):
    wrong_answer = []
    for index,value in enumerate(predicted):
        
        #print sign(value['position'])
        #print sign(actual[index].position)
        #print sign(value['position'])!=sign(actual[index].position)
        if sign(value['position'])!=sign(actual[index].position):
            wrong_answer.append((question_info[actual[index].questionId],(value['position'],actual[index].position)))
    return wrong_answer

#Returns dictionary of categories -> total occurances
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

#Returns dictionary of categories -> percent right, percent wrong, average questoin length, number of hints, average right position, average wrong position
def categories_right_wrong(actual,all):
    cat = {}
    for a in actual:
        key = question_info[a.questionId]['category']
        #print key
        if cat.has_key(key):
            cat[key][2]+=len(a.questionText.split())
            cat[key][3]+=total_hints(a.questionText)
            if a.position > 0:
                cat[key][0]+=1
                cat[key][4]+=a.position
            
            else:
                cat[key][1]+=1
                cat[key][5]+=a.position
        else:
            if a.position > 0:
                cat[key]=[1,0,len(a.questionText.split()),total_hints(a.questionText),
                          a.position,0]
            else:
                cat[key]=[1,0,len(a.questionText.split()),total_hints(a.questionText),
                          0,a.position]
    print_dictionary(cat)
    right = 0
    wrong = 0
    right_answer=0
    wrong_answer=0
    question_length=0
    hints = 0
    total = 0
    for (k,v) in cat.iteritems():
        cat[k] = [(v[0]/(v[0]+v[1]))*100,(v[1]/(v[0]+v[1]))*100,v[2]/all[k],
                  v[3]/all[k],v[4]/all[k],v[5]/all[k]]
        right+=v[0]
        wrong+=v[1]
        question_length+=v[2]
        hints+=v[3]
        right_answer+=v[4]
        wrong_answer+=v[5]
        total+=all[k]

    cat['TOTAL']=[(right/(right+wrong))*100,(wrong/(right+wrong))*100,question_length/total,
                  hints/total,right_answer/total,
                  wrong_answer/total]
                  
    return cat

#Returns dictionary of categories -> sign of answers don't match
def categories(wrong):
    cat = {}
    for w in wrong:
        if cat.has_key(w[0]['category']):
            cat[w[0]['category']]+=1
        else:
            cat[w[0]['category']]=1

    return cat

def categories_position_off(predicted,actual,all_cat):
    cat={}
    for index,value in enumerate(predicted):
        
        diff = abs(abs(value['position'])-abs(actual[index].position))
        if cat.has_key(actual[index].questionCategory):
            cat[actual[index].questionCategory]+=diff
        else:
            cat[actual[index].questionCategory]=diff

    for k,v in cat.iteritems():
        cat[k] = [v/all_cat[k]]

    return cat
        



#Returns dictionary of categories -> percent where sign of answer doesn't match
def percent_category_missed(wrong,total):
    percent_missed = {}
    for key,value in total.iteritems():
        if wrong.has_key(key):
            percent_missed[key] = [(wrong[key]/value)*100]
        else:
            percent_missed[key] = 0
    return percent_missed

#def length(wrong):

    


