# -*- coding: utf-8 -*-
"""
B351 - Project Chatbot

Project Team:
Tiana Deckard
Hans Thieme
David Bickel

"""

import numpy as np
import tensorflow as tf
import re
import random
import time
import sys

class ChatBot:
    
    def __init__(self,corpusTXT,corpusMAP):
        #load cornel corpus aquired from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/
        self.movieLines = open(corpusTXT, encoding='utf-8', errors='ignore').read().split('\n')
        self.convLines = open(corpusMAP, encoding='utf-8', errors='ignore').read().split('\n')
        #dictionary to hold ID to line mapping data
        self.dictId2Line = {}
        #list of conversations
        self.listConvs = []

        #load data into dictionary dictId2Line
        lineParts = []
        for line in self.movieLines:
            lineParts = line.split(' +++$+++ ')
            if (len(lineParts) == 5):
                self.dictId2Line[lineParts[0]] = lineParts[4]

        #load conversation data
        convParts = []
        for line in self.convLines[:-1]:
            #print ("Line: "+str(line))
            convParts = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            #print ("ConvParts: "+str(convParts))
            #convIds = convParts.split(',')
            #for convId in convIds:
            #    self.listConvs.append(convId)
            self.listConvs.append(convParts.split(','))
            #print ("listConvs: "+str(self.listConvs))

        #turn data into two lists questions and answers for training
        self.trainQuestions = []
        self.trainAnswers = []
        for c in self.listConvs:
            #print ("Convs:"+str(c))
            for i in range(len(c)-1):
                #print ("I val:"+str(i))
                lc1 = c[i]
                lc2 = c[i+1]
                #print ("keys: "+str(lc1)+" - "+ str(lc2))
                #print ("values: "+str(self.dictId2Line[lc1])+" - "+ str(self.dictId2Line[lc2]))
                
                self.trainQuestions.append(self.dictId2Line[lc1])
                self.trainAnswers.append(self.dictId2Line[lc2])
        print(len(self.trainQuestions))
        print(len(self.trainAnswers))

        #reduce set to sentences between a min and max of words
        min_senc_len = 2
        max_senc_len = 15

        reduce_qs_tmp = []
        reduce_ans_tmp = []
        reduce_qs = []
        reduce_ans = []

        i = 0
        for q in self.trainQuestions:
            if (len(q.split()) >= min_senc_len and len(q.split()) <= max_senc_len):
                reduce_qs_tmp.append(q)
                reduce_ans_tmp.append(self.trainAnswers[i])
                i += 1
        j = 0
        for a in reduce_ans_tmp:
            if (len(a.split()) >= min_senc_len and len(a.split()) <= max_senc_len):
                reduce_ans.append(a)
                reduce_qs.append(reduce_qs_tmp[j])
                j += 1

        #create vocabulary of words using dictionaries
        self.corpusWords = {}
        for q in reduce_qs:
            for w in q.split():
                if not (w in self.corpusWords):
                    self.corpusWords[w] = 1
                else:
                    self.corpusWords[w] += 1
            
        for a in reduce_ans:
            for w in a.split():
                if not (w in self.corpusWords):
                    self.corpusWords[w] = 1
                else:
                    self.corpusWords[w] += 1
        print("Distinct word count:", len(self.corpusWords))


    def converse(self,question):
        answer = "lol"

        #insert logic to pick proper response, for now just choose at random
        num = random.randint(0,50000)
        convToUse = self.movieLines[num]
        convParts = convToUse.split(' +++$+++ ')
        
        answer = self.dictId2Line[convParts[0]]
        
        return answer

    def train(self):
        return 1


#def main():
#   c = ChatBot()

#    print(str(len(c.movieLines)))
#    answer = c.converse("what")
#    print(str(answer)) 


#main()
