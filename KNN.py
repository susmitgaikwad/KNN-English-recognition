import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math

from scipy.sparse import hstack

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

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

pos_stuff = {} # find all unique POS tags and store in this list

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
# print(english_len)
		
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
# print(len(dict) - english_len)


## Here we're creating a vector of the POS tags for each 136 string sample
count = 0
tag_count = 0
length = {}
pos_vects = []

with open(pos_text_136_with_spaces_file, 'r') as f:
	for line in f:
		# print(line)
		arr = []		
		for tag in line.split():
			length.setdefault(len(line.split()), 0)
			length[len(line.split())] += 1
			if(tag not in pos_stuff):
					pos_stuff[tag] = tag_count
					tag_count += 1
			arr.append(tag)
		for i in range(42-len(line.split())): #42 is the longest sequence of POS tags
			# print(len(line.split()), 129-len(line.split()))
			arr.append('""')
		arr1 = " ".join(arr)
		pos_vects.append(arr1)
		count += 1
english_len = len(pos_vects)
# print(english_len)
		
with open(pos_word_salad_136_with_spaces_file, 'r') as f:
	for line in f:
		arr = []		
		for tag in line.split():
			length.setdefault(len(line.split()), 0)
			length[len(line.split())] += 1
			if(tag not in pos_stuff):
				pos_stuff[tag] = tag_count
				tag_count += 1
			arr.append(tag)
		for i in range(42-len(line.split())):
			# print(len(line.split()), 129-len(line.split()))
			arr.append('""')
		arr1 = " ".join(arr)
		pos_vects.append(arr1)
		count += 1
# print(len(pos_vects) - english_len)
# print(np.asarray(pos_vects).shape)

X = pd.DataFrame(dict, index = ['text']).T
X['pos_tags'] = pos_vects
print("Sample Size: ", X.shape[0])
print(X.head())
print("\n\nShuffling data...")


X['label'] = y
X = shuffle(X, random_state = 42)
y = X.iloc[:,-1]
print(X.head())
print()
print(y.head())
X.drop('label', axis = 1)

vectorizer = TfidfVectorizer()
X_1 = vectorizer.fit_transform(X['text']) # TF-IDF of 7-gram text
# print(type(X_1))

pos_vectorizer = TfidfVectorizer(analyzer='word', ngram_range = (2,2)) 
X_2 = pos_vectorizer.fit_transform(X['pos_tags']) # TF-IDF of POS tag bigrams

X = hstack([X_1, X_2]) # merging both matrices into one

y = pd.DataFrame(y)
print()
print("Shape of 7-gram matrix: ", X_1.shape)
print("Shape of POS bigram matrix: ", X_2.shape)
print("Total shape: ", X.shape)
# print(y.shape)
# print(y)

N = X.shape[0]
size = N

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel()[:size], test_size=0.33, random_state=42)
# print(sorted(sklearn.neighbors.VALID_METRICS['brute']))

def get_optimal_k(X_train, y_train):
	# creating odd list of K for KNN
	neighbors = [x for x in range(1, 100, 2)]

	# empty list that will hold cv scores
	cv_scores = []

	# perform 10-fold cross validation
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
		if(k%10 == 0):
			print("{}%".format(k), end= '>')
	print('Done')

	MSE = [1 - x for x in cv_scores]
	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print("The optimal number of neighbors is {}".format(optimal_k))

	# plot misclassification error vs k
	plt.plot(neighbors, MSE)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Misclassification Error')
	plt.show()
	return optimal_k
	
#
# kNN model training and testing
# 
# square = round(math.sqrt(X.shape[0])) # square root of the # of samples
# neighbors =  square # Number of neighbors we're using

# optimal_k = get_optimal_k(X_train, y_train)
optimal_k = 9
print()
print("Using {} neighbors...".format(optimal_k))
neigh = KNeighborsClassifier(n_neighbors=optimal_k)
neigh.fit(X_train, y_train)
predicted = neigh.predict(X_test)
print("Score: {}".format(accuracy_score(y_test, predicted)))
print()
print(classification_report(y_test, predicted, target_names=['english', 'salad']))

#
# ROC, AUC, and all that other stuff
#
probability = neigh.predict_proba(X_test)
preds = probability[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

print("Confusion Matrix:\n", confusion_matrix(y_test, predicted))
print_confusion_matrix(confusion_matrix(y_test, predicted), ['english', 'salad']).show()
plt.show()
