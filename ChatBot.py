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
    
    def __init__(self,corpusTXT,corpusMAP,chkPointPath):
        #load cornel corpus aquired from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/
        self.movieLines = open(corpusTXT, encoding='utf-8', errors='ignore').read().split('\n')
        self.convLines = open(corpusMAP, encoding='utf-8', errors='ignore').read().split('\n')
        #path to any previously loaded training data
        self.sessionPath = chkPointPath
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
        print ("The word 'pasta' appears in " + str(self.corpusWordOccurs['pasta'])+ " sentences in the reduced corpus.")

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
        print ("The word 'apple' appears in the answers: "+str(self.mapAnswers['apple']))

        self.train()


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

    def checkIfRestore(self, session, saver):
        #Restore the previously learned data from chatting
        chkpoint = tf.train.get_checkpoint_state(self.sessionPath)
        if (chkpoint and chkpoint.model_checkpoint_path):
            #restore previous state
            saver.restore(session, chkpoint.model_checkpoint_path)
        else:
            #start from scratch so no action needed
            print ("Starting from scratch..")

    def getSkipStep(self,iteration):
        #How long should we train before saving weights?
        if iteration < 100:
            return 30
        else: 
            return 100

    def loadPriorData(self, encFile, decFile, max_training_size=None):
        encodeFile = open(os.path.join(self.sessionPath, encFile), 'rb')
        decodeFile = open(os.path.join(self.sessionPath, decFile), 'rb')
        encode, decode = encodeFile.readline(), decodeFile.readline()
        dataBuckets = [[] for _ in self.cbm.buckets]
        i = 0
        while encode and decode:
            if (i + 1) % 10000 == 0:
                print("Bucket number", i)
            encodeIds = [int(id_n) for id_n in encode.split()]
            decodeIds = [int(id_n) for id_n in decode.split()]
            for bucketId, (encodeMaxSize, decodeMaxSize) in enumerate(self.cbm.buckets):
                if len(encodeIds) <= encodeMaxSize and len(decodeIds) <= decodeMaxSize:
                    dataBuckets[bucketId].append([encodeIds, decodeIds])
                    break
            encode, decode = encodeFile.readline(), decodeFile.readline()
            i += 1
        return dataBuckets

    def getBuckets(self):
        # Build buckets as a data set we can use for our calculations
        self.testBuckets = self.loadPriorData("testBuckets.enc", "testBuckets.dec")
        self.dataBuckets = self.loadPriorData("trainBuckets.enc", "trainBuckets.dec")
        trainingBucketSizes = [len(self.dataBuckets[b]) for b in range(len(self.cbm.buckets))]
        print("Number of samples in a bucket: "+str(trainingBucketSizes))
        trainingTotalSize = sum(trainingBucketSizes)
        if trainingTotalSize < 1:
            trainingTotalSize = 1
        # Build a list of increasing numbers from 0 to 1 that will be used in bucket selection
        self.bucketScale = [sum(trainingBucketSizes[:i + 1]) / trainingTotalSize
                       for i in range(len(trainingBucketSizes))]
        print("Bucket scale: "+str(self.bucketScale))
        return self.testBuckets, self.dataBuckets, self.bucketScale

    def getRandomBucket(self,bucketScale):
        r = random.random()
        print ("BucketScale: "+str(bucketScale))
        return min([i for i in range(len(bucketScale))
                    if bucketScale[i] > r])
    
    def checkLengths(self,encoderSize, decoderSize, encInputs, decInputs, decMasks):
        #check that the encoder inputs, decoder inputs, and decoder masks are of the expected lengths
        if len(encInputs) != encoderSize:
            raise ValueError("Encoder length must be equal to that of the provided bucket,"
                            " %d != %d." % (len(encInputs), encoderSize))
        if len(decInputs) != decoderSize:
            raise ValueError("Decoder length must be equal to that of the provided bucket,"
                           " %d != %d." % (len(decInputs), decoderSize))
        if len(decMasks) != decoderSize:
            raise ValueError("Weights length must be equal to the one provided by the bucket,"
                           " %d != %d." % (len(decMasks), decoderSize))

    def processStep(self,session, model, encInputs, decInputs, decMasks, bucketId, direction):
        #Process a single step of the training..
        #use direction value of "forward" when you are only wanting to train in one direction
        # or just chatting with the bot
        encoderSize, decoderSize = self.cbm.buckets[bucketId]
        checkLengths(encoderSize, decoderSize, encInputs, decInputs, decMasks)

        # setup an input feed from encoder and decoder data
        inputFeed = {}
        for step in range(encoderSize):
            inputFeed[self.cbm.encInputs[step].name] = encInputs[step]
        for step in range(decoderSize):
            inputFeed[self.cbm.decInputs[step].name] = decInputs[step]
            inputFeed[self.cbm.decMasks[step].name] = decMasks[step]

        lastTarget = self.cbm.decInputs[decoderSize].name
        inputFeed[lastTarget] = np.zeros([self.cbm.batchSize], dtype=np.int32)#??

        # output feed: depends on whether we do a backward step or not.
        if not (direction == "forward"):
            outputFeed = [self.cbm.trainOps[bucketId],  #update opertion
                           self.cbm.gradientNorms[bucketId],  #gradient norm
                           self.cbm.losses[bucketId]]  #loss
        else:
            outputFeed = [self.cbm.losses[bucket_id]]  # loss
            for step in range(decoderSize):  # output logits.
                outputFeed.append(self.cbm.outputs[bucketId][step])

        outputs = session.run(outputFeed, inputFeed)
        if not (direction == "forward"):
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def getSingleBatch(self,data_bucket, bucketId, batch_size=1):
        # Return a single batch for the model
        encoderSize, decoderSize = self.cbm.buckets[bucketId]
        encoder_inputs, decoder_inputs = [], []

        for _ in range(batch_size):
            encInput, decInput = random.choice(data_bucket)
            # pad data for both encoder and decoder, but reverse the encoder
            encInputs.append(list(reversed((encInput + (0 * (encoderSize - len(encInput)))))))
            decInputs.append(decInput + (0 * (decoderSize - len(decInput))))

        # now we create batch-major vectors from the data selected above.
        batchEncoderInputs = reshape_batch(encoder_inputs, encoder_size, batch_size)
        batchDecoderInputs = reshape_batch(decoder_inputs, decoder_size, batch_size)

        # create decoder_masks to be 0 for decoders that are padding.
        batchMasks = []
        for length_id in range(decoder_size):
            batch_mask = np.ones(batch_size, dtype=np.float32)
            for batch_id in range(batch_size):
                # we set mask to 0 if the corresponding target is a PAD symbol.
                # the corresponding decoder is decoder_input shifted by 1 forward.
                if length_id < decoder_size - 1:
                    target = decoder_inputs[batch_id][length_id + 1]
                if length_id == decoder_size - 1 or target == config.PAD_ID:
                    batch_mask[batch_id] = 0.0
            batch_masks.append(batch_mask)
        return batch_encoder_inputs, batch_decoder_inputs, batch_masks

    def evalTestData(self,session, chatmodel, testBuckets):
        #run evaluation on testing data set
        for bucketId in range(len(self.cbm.buckets)):
            if len(testBuckets[bucketId]) == 0:
                print("Testing empty bucket %d" % (bucketId))
                continue
            startTime = time.time()
            encInputs, decInputs, decMasks = data.getSingleBatch(testBuckets[bucketId], 
                                                                        bucketId,
                                                                        batch_size=self.cbm.batchSize)
            _, stepLoss, _ = processStep(session, chatmodel, encInputs, decInputs, 
                                   decoder_masks, bucket_id, True)
            print('Test bucket {}: loss {}, time {}'.format(bucketId, stepLoss, time.time() - startTime))


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

            cap = []
            low = []
        
            for i in range(len(q_p)):
                #if word exists in the corpus it can be relavant, ignore ones that would throw an error
                print ("q_p at i:"+str(q_p[i]))
                if (q_p[i] in self.mapAnswers and q_p[i] != ""):
                    cap = relTerm[:1].upper() + relTerm[1:]
                    low = relTerm[:1].lower() + relTerm[1:]
                    
                    tf = self.calc_tf(question,q_p[i])
                    idf = self.calc_idf(q_p[i])
                    if((tf * idf) > tfIdf):
                        tfIdf = (tf * idf)
                        relTerm = q_p[i]
                i += 1

            #get sentences that contain the relevent term
            #get sentences that contain the relevent term
            potentialAnswersC = self.mapAnswers[cap] #capitals
            potentialAnswersL = self.mapAnswers[low] #lowers

            potentialAnswers = []

            for p in potentialAnswersC:
                potentialAnswers.append(p)

            for p in potentialAnswersL:
                potentialAnswers.append(p)

            #choose at random for now
            answer_index = random.randint(0,len(potentialAnswers)-1)
            num = potentialAnswers[answer_index]
        
            #convToUse = self.movieLines[num]
            #convParts = convToUse.split(' +++$+++ ')
            #answer = self.dictId2Line[convParts[0]]

            #for e in potentialAnswers:
                #print(self.reduce_ans[e])
        
            answer = self.reduce_ans[num]
            ans_p = answer.split()
            tf = self.calc_tf(answer,ans_p[0])
            idf = self.calc_idf(ans_p[0])
            print ("TF and IDF "+str(tf)+" - "+str(idf))
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
            answer = auto_replies[answer_index]

        return answer

    def train(self):

        #build buckets and graph from our data to use for chat
        self.testingBuckets, self.dataBuckets, self.bucketScale = self.getBuckets()
        self.cbm.buildGraph()
        
        #create a saver to store session information for the next run
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Starting Training session..")
            sess.run(tf.global_variables_initializer())

            self.checkIfRestore(sess, saver)

            iteration = self.cbm.globalStep.eval()
            totalLoss = 0
            while True:
                skipStep = self.getSkipStep(iteration)
                bucketId = self.getRandomBucket(self.bucketScale)
                encInputs, decInputs, decMasks = self.getSingleBatch(self.dataBuckets[bucketId],
                                                                bucketId,
                                                                batch_size=self.cbm.batchSize)
                startTime = time.time()
                _, stepLoss, _ = self.processStep(sess, self.cbm, encInputs, decInputs, decMasks, bucketId, False)
                totalLoss += stepLoss
                iteration += 1
                if (iteration % skipStep == 0):
                    print("Iteration {}: loss {}, time {}".format(iteration, totalLoss/skipStep, time.time() - startTime))
                    startTime = time.time()
                    totalLoss = 0
                    saver.save(sess, os.path.join(self.sessionPath, 'chatbot'), global_step=self.cbm.globalStep)
                    if (iteration % (10 * skipStep) == 0):
                        # Run evals on development set and print their loss
                        self.evalTestData(sess, model, test_buckets)
                        startTime = time.time()
                    sys.stdout.flush()
        return 1


#def main():
#   c = ChatBot()

#    print(str(len(c.movieLines)))
#    answer = c.converse("what")
#    print(str(answer)) 


#main()
