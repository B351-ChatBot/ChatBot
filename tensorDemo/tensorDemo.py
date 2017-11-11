# -*- coding: utf-8 -*-
"""
B351 - Project Chatbot
TensorFlow demo from http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

Tensor flow requires the 64 bit version of Python
"""

import tensorflow as tf

def main():
    #perform (d=b+c and e=c+2) 
    const = tf.constant(2.0,name="const")

    b = tf.Variable(2.0,name='b')
    c = tf.Variable(1.0, name='c')
    
    #operations
    d = tf.add(b,c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d,e,name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # create TensorFlow variables
        #b = tf.placeholder(tf.float32, [None, 1], name='b')
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a)
        #a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
        print("Variable a is {}".format(a_out))



    
    
            
main()
    
