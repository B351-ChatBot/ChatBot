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
import os
import random
import math
import time
import sys
from ChatBotModel import *

class ChatBot:
    
    def __init__(self,corpusTXT,corpusMAP):
        #load cornel corpus aquired from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/
        self.movieLines = open(corpusTXT, encoding='utf-8', errors='ignore').read().split('\n')
        self.convLines = open(corpusMAP, encoding='utf-8', errors='ignore').read().split('\n')

        #dictionary to hold ID to line mapping data
        self.dictId2Line = {}
        #list of conversations
        self.listConvs = []
        self.cbm = ChatBotModel(direction="forward", size=1)

        #load data into dictionary dictId2Line
        lineParts = []
        for line in self.movieLines:
            lineParts = line.split(' +++$+++ ')
            if (len(lineParts) == 5):
                self.dictId2Line[lineParts[0]] = lineParts[4]

        #load conversation data
        convParts = []
        for line in self.convLines[:-1]:
            convParts = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            self.listConvs.append(convParts.split(','))

        #turn data into two lists questions and answers for training
        self.trainQuestions = []
        self.trainAnswers = []
        for c in self.listConvs:

            for i in range(len(c)-1):
                lc1 = c[i]
                lc2 = c[i+1]
                self.trainQuestions.append(self.dictId2Line[lc1])
                self.trainAnswers.append(self.dictId2Line[lc2])

        #reduce set to sentences between a min and max of words
        min_senc_len = 1
        max_senc_len = 15

        reduce_qs_tmp = []
        reduce_ans_tmp = []
        self.reduce_qs = []
        self.reduce_ans = []

        i = 0
        for q in self.trainQuestions:
            if (len(q.split()) >= min_senc_len and len(q.split()) <= max_senc_len):
                reduce_qs_tmp.append(q)
                reduce_ans_tmp.append(self.trainAnswers[i])
                i += 1
        j = 0
        for a in reduce_ans_tmp:
            if (len(a.split()) >= min_senc_len and len(a.split()) <= max_senc_len):
                self.reduce_ans.append(a)
                self.reduce_qs.append(reduce_qs_tmp[j])
                j += 1

        print ("Count of reduced questions: " +str(len(self.reduce_qs))+" - answers: "+str(len(self.reduce_ans)))
        #create vocabulary of words using dictionaries
        self.corpusWords = {}
        for q in self.reduce_qs:
            for w in q.split():
                wlow = w.lower()
                if not (wlow in self.corpusWords):
                    self.corpusWords[wlow] = 1
                else:
                    self.corpusWords[wlow] += 1
            
        for a in self.reduce_ans:
            for w in a.split():
                wlow = w.lower()
                if not (wlow in self.corpusWords):
                    self.corpusWords[wlow] = 1
                else:
                    self.corpusWords[wlow] += 1
        print("Distinct word count:", len(self.corpusWords))
        print ("The word 'the' is used " + str(self.corpusWords['the'])+ " times.")

        #create an inverse dictionary to map count to a word
        self.corpusCountToWords = {value: key for key, value in self.corpusWords.items()}
        print ("A word that is used 7 times is " + str(self.corpusCountToWords[7])+ ".")

        #create a dictionary to map count of sentences containing certain words
        self.corpusWordOccurs = {}
        for q in self.reduce_qs:
            qList = q.split()
            qList = list(set(q))
            for w in qList:
                wlow = w.lower()
                if not (wlow in self.corpusWordOccurs):
                    self.corpusWordOccurs[wlow] = 1
                else:
                    self.corpusWordOccurs[wlow] += 1
            
        for a in self.reduce_ans:
            aList = a.split()
            aList = list(set(aList))
            for w in aList:
                wlow = w.lower()
                if not (wlow in self.corpusWordOccurs):
                    self.corpusWordOccurs[wlow] = 1
                else:
                    self.corpusWordOccurs[wlow] += 1
        print("Word length of the Word Occurance object is:", len(self.corpusWordOccurs))
        print ("The word 'pasta' appears in " + str(self.corpusWordOccurs['pasta'])+ " sentences in the reduced corpus.")

        #build dictionary of each word with a unique index value in the cleaned up questions and answers, also the reverse for index -> word
        self.corpusWordToIndex = {}
        i = 0
        for q in self.reduce_qs:
            qList = q.split()
            qList = list(set(q))
            for w in qList:
                wlow = w.lower()
                if not (wlow in self.corpusWordToIndex):
                    self.corpusWordToIndex[wlow] = i
                    i+=1
        for a in self.reduce_ans:
            aList = a.split()
            aList = list(set(aList))
            for w in aList:
                wlow = w.lower()
                if not (wlow in self.corpusWordToIndex):
                    self.corpusWordToIndex[wlow] = i
                    i+=1
        self.corpusIndexToWord = {}
        for w in self.corpusWordToIndex:
            i = self.corpusWordToIndex[w]
            self.corpusIndexToWord[i] = w
        
        print("The length of the Word Index object is:", len(self.corpusWordToIndex))
        print ("The word 'pasta' has an index value of " + str(self.corpusWordToIndex['pasta'])+ ", and the word 'sauce' has an index of " + str(self.corpusWordToIndex['sauce']))

        #build dictionary of words -> conversations/sentences
        self.mapAnswers = {}
        i = 0
        for i in range(len(self.reduce_ans)):
            ans = self.reduce_ans[i]
            aList = ans.split()
            aList = list(set(aList))
            for w in aList:
                wlow = w.lower()
                if not (wlow in self.mapAnswers):
                    self.mapAnswers[wlow] = [i]
                else:
                    self.mapAnswers[wlow].append(i)
            i += 1
        print ("The word 'apple' appears in the answers: "+str(self.mapAnswers['apple']))

        #show some basic stats from initialization
        print ("Total words in the corpus: "+str(len(self.corpusWordToIndex)))
        print ("Some info for the word 'orange'\nindex: "
               +str(self.corpusWordToIndex['orange'])
               +"\noccurs: "
               +str(self.corpusWords['orange'])
               +" \nexists in # of sentences: "
               +str(len(self.mapAnswers['orange'])))


    def calc_tf(self,sentence,word):
        frequencies = {}
        wCount = 0
        for w in sentence.split():
            if not (w in frequencies):
                frequencies[w] = 1
            else:
                frequencies[w] += 1
            wCount += 1
        tf = frequencies[word] / wCount
        return tf

    def calc_idf(self,word):
        bigN = len(self.reduce_qs)
        littleN = self.corpusWordOccurs[word]
        idf = math.log(bigN/littleN)
        return idf

    def cleanS(self,statement):
        statement = statement.replace("?","")
        statement = statement.replace(".","")
        statement = statement.replace(",","")
        statement = statement.replace("!","")
        statement = statement.replace(":","")
        statement = statement.replace("&","")
        return statement


    def converse(self,question):
        #clean up the question text
        question = self.cleanS(question)
        #if question contains words for analysis
        if (question != ""):
            print ("Q: "+question)
            #determine most relavant term in given statement
            q_p = question.split()
            tfIdf = 0
            i = 0
            relTerm = q_p[0]
            low = relTerm.lower()
        
            for i in range(len(q_p)):
                #if word exists in the corpus it can be relavant, ignore ones that would throw an error
                low = q_p[i].lower()
                qlow = question.lower()
                print ("Current word: "+low)
                if (low in self.mapAnswers and low != ""):

                    print ("Previous Relevant Word: "+str(relTerm))
                    print ("Previous TF-IDF Score: "+str(tfIdf))                    
                    tf = self.calc_tf(qlow,low)
                    idf = self.calc_idf(low)
                    newTfIdf = (tf * idf)
                    print ("New TF-IDF Score: "+str(newTfIdf)) 
                    if(newTfIdf > tfIdf):
                        tfIdf = newTfIdf
                        relTerm = low
                i += 1

            lowRelTerm = relTerm.lower()

            potentialAnswersL = self.mapAnswers[lowRelTerm] #lowers

            potentialAnswers = []

            for p in potentialAnswersL:
                potentialAnswers.append(p)

            #instead of random sentence selection lets use highest tf-idf to make selection
            bestTfIdf = 0
            for answer in potentialAnswers:
                answerText = self.reduce_ans[answer]
                ans_tf = self.calc_tf(answerText.lower(),lowRelTerm)
                ans_idf = self.calc_idf(lowRelTerm)
                ans_tfIdf = (tf * idf)
                if(ans_tfIdf >= bestTfIdf):
                    top_answer = answerText
                print ("Answer "+str(top_answer))
                print ("Answer TF-IDF "+str(tfIdf))
            
        else:
            #reply with confusion as not valid question was provided
            auto_replies = ["Dave, this conversation can serve no purpose anymore. Goodbye.",
                            "42",
                            "I'm sorry, Dave. I'm afraid I can't do that.",
                            "I didn't get that",
                            "who me?",
                            "what dat?"
                            "I am not sure I know what you mean",
                            "I am not programmed to help you with that"]
            #replies pull from 2001 space odyssey, hitchhikers guide to the galaxy, and siri/alexa replies 
            answer_index = random.randint(0,len(auto_replies)-1)
            top_answer = auto_replies[answer_index]

        return top_answer

    def train(self):


        return 1
