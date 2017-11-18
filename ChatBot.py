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
import math
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
                if not (w in self.corpusWords):
                    self.corpusWords[w] = 1
                else:
                    self.corpusWords[w] += 1
            
        for a in self.reduce_ans:
            for w in a.split():
                if not (w in self.corpusWords):
                    self.corpusWords[w] = 1
                else:
                    self.corpusWords[w] += 1
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
                if not (w in self.corpusWordOccurs):
                    self.corpusWordOccurs[w] = 1
                else:
                    self.corpusWordOccurs[w] += 1
            
        for a in self.reduce_ans:
            aList = a.split()
            aList = list(set(aList))
            for w in aList:
                if not (w in self.corpusWordOccurs):
                    self.corpusWordOccurs[w] = 1
                else:
                    self.corpusWordOccurs[w] += 1
        print("Word length of the Word Occurance object is:", len(self.corpusWordOccurs))
        print ("The word 'Jennifer' appears in " + str(self.corpusWordOccurs['Jennifer'])+ " sentences in the reduced corpus.")

        #build dictionary of words -> conversations/sentences
        self.mapAnswers = {}
        i = 0
        for i in range(len(self.reduce_ans)):
            ans = self.reduce_ans[i]
            #print ("A:"+ans)
            aList = ans.split()
            aList = list(set(aList))
            for w in aList:
                if not (w in self.mapAnswers):
                    self.mapAnswers[w] = [i]
                else:
                    self.mapAnswers[w].append(i)
            i += 1
        print ("The word 'Jennifer' appears in the answers: "+str(self.mapAnswers['Jennifer']))

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
        print ("Q: "+question)
        #determine most relavant term in given statement
        q_p = question.split()
        tfIdf = 0
        i = 0
        relTerm = q_p[0]
        for i in range(len(q_p)):
            tf = self.calc_tf(question,q_p[i])
            idf = self.calc_idf(q_p[i])
            if((tf * idf) > tfIdf):
                tfIdf = (tf * idf)
                relTerm = q_p[i]
            i += 1

        #get sentences that contain the relevent term
        potentialAnswers = self.mapAnswers[relTerm]

        #choose at random for now
        answer_index = random.randint(0,len(potentialAnswers)-1)
        num = potentialAnswers[answer_index]
        
        #convToUse = self.movieLines[num]
        #convParts = convToUse.split(' +++$+++ ')
        #answer = self.dictId2Line[convParts[0]]
        
        answer = self.reduce_ans[num]
        ans_p = answer.split()
        tf = self.calc_tf(answer,ans_p[0])
        idf = self.calc_idf(ans_p[0])
        print ("TF and IDF "+str(tf)+" - "+str(idf))
        return answer

    def train(self):
        return 1


#def main():
#   c = ChatBot()

#    print(str(len(c.movieLines)))
#    answer = c.converse("what")
#    print(str(answer)) 


#main()
