import numpy as np
import pandas as pd
import sklearn

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

file_dir = './files'
space_dir = file_dir+'/spaces/'
no_space_dir = file_dir+'/nospaces/'
ngram_dir = file_dir+'/ngrams/'
pos_dir = file_dir+'/pos/'

input_corpus_file = 'alice_oz.txt'
english_text_136_with_spaces_file = space_dir+'english_text_136_with_spaces.txt'
english_text_136_without_spaces_file = no_space_dir+'english_text_136_without_spaces.txt'
pos_text_136_with_spaces_file = pos_dir+'pos_text_136_with_spaces.txt'
n_grams_136_english_text_file = ngram_dir+'n_grams_136_english_text.txt'

junk1_file = 'junk1.txt'
word_salad_136_with_spaces_file = space_dir+'word_salad_136_with_spaces.txt'
word_salad_136_without_spaces_file = no_space_dir+'word_salad_136_without_spaces.txt'
pos_word_salad_136_with_spaces_file = pos_dir+'pos_word_salad_136_with_spaces.txt'
n_grams_136_word_salad_file = ngram_dir+'n_grams_136_word_salad.txt'

df = pd.DataFrame()

dict = {}
y = []
count = 0
length = {}
corpus = []

pos_tag = {
	'NN' : 1,
	'DT' : 2,
	'JJ' : 3,
	'PR' : 4, 
	'RB' : 5,
	'VB' : 6,
	'IN' : 7 
}




with open(n_grams_136_english_text_file, 'r') as f:
	for line in f:
		# print(line)
		arr = []		
		for gram in line.split():
			length.setdefault(len(line.split()), 0)
			length[len(line.split())] += 1
			arr.append(gram)
		for i in range(129-len(line.split())):
			# print(len(line.split()), 129-len(line.split()))
			arr.append('')
		# print(len(arr))
		# for gram in arr:
			# print(gram)		
		arr1 = " ".join(arr)
		corpus.append(arr1)
		y.append(0)
		dict[count] = arr1
		count += 1
english_len = len(dict)
print(english_len)
		
with open(n_grams_136_word_salad_file, 'r') as f:
	for line in f:
		arr = []		
		for gram in line.split():
			arr.append(gram)
		for i in range(129-len(line.split())):
			# print(len(line.split()), 129-len(line.split()))
			arr.append('')
		arr1 = " ".join(arr)
		corpus.append(arr1)
		y.append(1)
		dict[count] = arr1
		
		count += 1
print(len(dict) - english_len)


# print(corpus[:4])
# import re
# dict = {}
# with open(english_text_136_without_spaces_file, 'r') as f:
	# for line in f:
		# dict[count] = line
		# count += 1

# def ngrams(string, n=3):
    # string = re.sub(r'[,-./]|\sBD',r'', string)
    # ngrams = zip(*[string[i:] for i in range(n)])
    # return [''.join(ngram) for ngram in ngrams]

# print(dict)
# print(ngrams(text, n=7))		

'''
corpus = []

tfidf_vectorizer = TfidfVectorizer(norm=None, ngram_range=(2,2))
with open(english_text_136_with_spaces_file, 'r') as f:
	for line in f:
		corpus.append(line)
with open(word_salad_136_with_spaces_file, 'r') as f:
	for line in f:
		corpus.append(line)

		
new_term_freq_matrix = tfidf_vectorizer.fit_transform(corpus)
print(tfidf_vectorizer.vocabulary_)
'''	
# print(y)
X = pd.DataFrame(dict, index = ['text']).T
print(X.head())


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X['text'])
# print(vectorizer.get_feature_names())
print(X.shape)

y = pd.DataFrame(y)
# print(X)
print(X.shape)
print(y.shape)
# print(y)

N = X.shape[0]
size = N
X_train, X_test, y_train, y_test = train_test_split(X[:size], y.values.ravel()[:size], test_size=0.33, random_state=42)
# print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
dist = DistanceMetric.get_metric('jaccard')
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
predicted = neigh.predict(X_test)
print("Score: {}%".format(accuracy_score(y_test, predicted)))
print(classification_report(y_test, predicted))