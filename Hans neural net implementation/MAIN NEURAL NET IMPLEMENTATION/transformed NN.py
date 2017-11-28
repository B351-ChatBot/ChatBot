#PESUDOCODE

#1: CREATING DATA SETS AND LABELS #>>> use create_sentiment_featuresets.py for reference for next chunk

    # https://www.youtube.com/watch?v=7fcWfUavO7E, DOWNLOAD NLTK, 6:00
    #use nltk
    ##import nltk
    ##from nltk.tokenize import word_tokenize
    ##from nltk.stem import WordNetLemmatizer

    #from corpus, create a list of all words:
    #use tolkenize to turn the string of all words into, list of all words.
    #allWords = word_tolkenize
    #stem the list of all words, then lemmatizer. use create_sentiment_featuresets.py for reference
    #create
    #freqDict = {"the":241251, "all":41241, ...

    #trim to a list, entries in freqDict with freq about <1000? so arbitrarily rare words only, this becomes
    #   InputVector = ["apple", "politics", ...

    #
    
#2: INITIALIZING THE NEURAL NET

    #requires a list of all rare unique words, and a list of all unique answers. e.g. work in 1: first
    #need to convert the output to return top 10....


#3: TRAINING

    #Input = [[0 1 0 0 1 ...x] correct reply, [0 ...] y] where x is the number of rare unique words, and y is the number of questions
    #correct reply = [0 0 0 ... 1 .. 0 .. x] wjere x = number of unique answers.
    #this is stored in a pickle in original example

    #returns 10 best guesses in response
    #if true response isnt in best guesses,

#4: Interaction

    #need to figure out how to interact with neural net after training
    #need to place it within the proper frameworks in our code









from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):
    #print(data): some placeholder? 

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    #print(output): #Tensor("add:0", shape=(?, 2), dtype=float32), only called once. presumably to make the model, tf does the rest
    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
                                                                        #print(train_y[start]) #[1,0], aka the label, trainx = input
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				t, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})  #two operations, defined above
				#print(t) #optimizer spit out none
##				print(feed_dict) #not defined, prob an argument being fed into both optimizer then cost.

##				print(sess.run(tf.nn.top_k(train_x[start], 10)))
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
		

	    
train_neural_network(x)
