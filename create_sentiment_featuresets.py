import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):

	lexicon = []
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l) #tokenize takes a str, returns a list of words
			lexicon += list(all_words)

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	#stem + lemmatize removes tense, plural
	w_counts = Counter(lexicon)
	#w_counts = {"the":52152, ...}
	l2 = []
	for w in w_counts: #limits words to rare, but not extremely rare words
		#print(w_counts[w])
		if 1000 > w_counts[w] > 10:
			l2.append(w)
	print(len(l2))
	return l2  #the input into NN





def sample_handling(sample,lexicon,classification):

	featureset = []
	#featureset will eventually =
##	[
##                [[0 1 0 1...], [1 0...]]
##                [[feature], [label]], feature is input into NN, label is correct answer
##      ] data input on left, correct answer on right, in each cell

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset



def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	#features += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features) 
	features = np.array(features)

	testing_size = int(test_size*len(features)) #test size = last 10% or .1

	#this dogs is ok
	#dog ok,
        #[0,0,0,0,0,0..... 1, ... 1, ..]

	
	#TRAIN_X[2] "LABEL" OR ANSWER IS TRAIN_Y[2]

	train_x = list(features[:,0][:-testing_size]) # features[:,0] = features, [1 0 0 1... representing rare unique words in question.
                #train_x is the list of every question, formatted into [0 1 0 0..] form. where a 1 represents it has the corrosopnding rare word, 0 
	train_y = list(features[:,1][:-testing_size]) # features[:,1] = labels, [0 0 0 0... 1 ... 0] representing the one right answer
                #train_y is the list of every answer, formatted into [0 1 0 0..] form. where a 0 or 1 represents a whole answer

	
	test_x = list(features[:,0][-testing_size:]) #final accuracy test, using last 10% of data
	test_y = list(features[:,1][-testing_size:]) 

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	# if you want to pickle this data: pickle = save data
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)

