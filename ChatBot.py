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

class ChatBot:
    
    def __init__(self,corpusTXT,corpusMAP):
        #load cornel corpus aquired from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/
        self.movieLines = open(corpusTXT, encoding='utf-8', errors='ignore').read().split('\n')
        self.convLines = open(corpusMAP, encoding='utf-8', errors='ignore').read().split('\n')
        #dictionary to hold ID to line mapping data
        self.dictId2Line = {}
        #list of conversations
        self.listConv = []

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
            self.listConv.append(convParts.split(','))

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
