"""
B351 - Project Chatbot

Project Team:
Tiana Deckard
Hans Thieme
David Bickel

A model to use tensorflow for deep text learning based on Google Translate
Tensorflow model https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
using Sequence to sequence model developed by Cho (2014) & pulling heavily from an example
by Chip Huyen from Stanford titled "TensorFlow for Deep Learning Research" 

"""

import time
import numpy as np
import tensorflow as tf

class ChatBotModel(object):
    def __init__(self, direction, size):
        print('Initialize a new ChatBotModel..')

        self.direction = direction
        self.batchSize = size
        self.buckets = [(16, 19)]
        self.layers = 3
        self.hiddenNetSize = 256
        self.batchSize = 64
        self.LR = 0.5 #loss ratio
        self.maxGradNorm = 5.0
        self.sampleSize = 512
        self.decWords = 1000#?
        self.encWords = 1000#?
    
    def createPlaceholders(self):
        # create placeholders for tensorflow
        print('Creating placeholders..')
        self.encInputs = [tf.placeholder(tf.int32,
                                         shape=[None],
                                         name='encoder{}'.format(i))
                          for i in range(self.buckets[-1][0])]
        self.decInputs = [tf.placeholder(tf.int32,
                                         shape=[None],
                                         name='decoder{}'.format(i))
                          for i in range(self.buckets[-1][1]+1)]
        self.decMasks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(self.buckets[-1][1]+1)]

        # Our targets are the decoder inputs shifted over by one??
        self.targets = self.decInputs[1:]
        
    def createInference(self):
        print('Creating inference..')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size??
        if self.sampleSize > 0 and self.sampleSize < self.decWords:
            self.w = tf.get_variable('proj_w', [self.hiddenNetSize, self.decWords])
            self.b = tf.get_variable('proj_b', [self.decWords])
            self.outputProjection = (self.w, self.b)
            
        def sampleLoss(labels, logits):
            labels = tf.reshape(labels, [-1, 1]) #??
            return tf.nn.sampled_softmax_loss(tf.transpose(self.w),
                                              self.b,
                                              labels,
                                              logits,
                                              self.sampleSize,
                                              self.decWords)
            
        self.softmaxLossFunction = sampleLoss

        singleCell = tf.contrib.rnn.GRUCell(self.hiddenNetSize)
        self.cell = tf.contrib.rnn.MultiRNNCell([singleCell] * self.layers)

    def createLoss(self):
        print("Creating loss... ")
        startTime = time.time()
        def seq2seqFunction(encInputs, decInputs, doDecode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encInputs,
                                                                         decInputs,
                                                                         self.cell,
                                                                         num_encoder_symbols=self.encWords,
                                                                         num_decoder_symbols=self.decWords,
                                                                         embedding_size=self.hiddenNetSize,
                                                                         output_projection=self.outputProjection,
                                                                         feed_previous=doDecode)

        if self.direction == "forward":
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encInputs,
                                                                         self.decInputs,
                                                                         self.targets,
                                                                         self.decMasks,
                                                                         self.buckets,
                                                                         lambda x, y: seq2seqFunction(x, y, True),
                                                                         softmax_loss_function=self.softmaxLossFunction)
            # If we use output projection, we need to project outputs for decoding??
            if self.outputProjection:
                for bucket in range(len(self.buckets)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.outputProjection[0]) + self.outputProjection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encInputs,
                                                                         self.decInputs,
                                                                         self.targets,
                                                                         self.decMasks,
                                                                         self.buckets,
                                                                         lambda x, y: seq2seqFunction(x, y, False),
                                                                         softmax_loss_function=self.softmaxLossFunction)
        print("Time Taken:", time.time() - startTime)

    def createOptimizer(self):
        print("Creating optimizer... ")
        with tf.variable_scope('training') as scope:
            self.globalStep = tf.Variable(0, dtype=tf.int32, trainable=False, name='globalStep')

            if not (self.direction == "forward"):
                self.optimizer = tf.train.GradientDescentOptimizer(self.LR)
                trainableVars = tf.trainable_variables()
                self.gradientNorms = []
                self.trainOps = []
                startTime = time.time()
                for bucket in range(len(self.buckets)):
                    clippedGrads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainableVars),
                                                                 self.MaxGradNorm)
                    self.gradientNorms.append(norm)
                    self.trainOps.append(self.optimizer.apply_gradients(zip(clippedGrads, trainableVars), 
                                                            global_step=self.globalStep))
                    print('Creating the optimization for bucket {} took {} seconds'.format(bucket, time.time() - startTime))
                    startTime = time.time()


    def createSummary(self):
        return 1

    def buildGraph(self):
        self.createPlaceholders()
        self.createInference()
        self.createLoss()
        self.createOptimizer()
        self.createSummary()
