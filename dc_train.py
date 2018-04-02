# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 13:23:20 2018

@author: Shashankar
"""


import csv
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing


# pre-processing 
with open(r'C:\Users\Shashankar\Desktop\HeavyWater\shuffled-full-set-hashed.csv','rb') as f:
    reader = csv.reader(f)
    title = [] # this is the list of labels
    content = [] # this has the actual contents of the document
    for row in reader:
        title.append(row[0])# file consists of only 2 columns - one for label and the other content
        content.append(row[1])
labels = list(set(title)) # getting the set of labels
le = preprocessing.LabelEncoder() # python's label encoder to convert to numerical format
le.fit(labels)
num_labels = le.transform(title)
train_len = 50000 # setting training set to be roughly ~80% of the total amount

Y_train,Y_test=num_labels[:train_len],num_labels[train_len:] # dividing the labels into train and test
# the code below mostly uses models from scikit-learn
cv = CountVectorizer() # to get the frequency of the various words
cv.fit_transform(content) 
word_freq_matrix = cv.transform(content)
tfidf = TfidfTransformer(norm="l2") # implements a term frequency-inverse document frequency model
tfidf.fit(word_freq_matrix)
tf_idf_matrix = tfidf.transform(word_freq_matrix) # we get the tf-idf matrix with appropriate mappings

X_train,X_test=tf_idf_matrix[:train_len],tf_idf_matrix[train_len:] # divide the input data(content) into train and test
logreg = linear_model.LogisticRegression(C=1e5) # implement a logistic regression model with inverse reg strength = 1e5
logreg.fit(X_train,Y_train)
pred=logreg.predict(X_test)
accuracy_score(Y_test, pred) # checking accuracy
with open("dc_model.pkl", "wb") as f:
    pickle.dump([logreg, tfidf, cv], f) # dumping our model so that we lambda can use directly

#predict for a particular case (sanity check for lambda implementation)
document = r"C:\Users\Shashankar\Desktop\test.txt"
docu_file = open(document, 'r')
text = [docu_file.readline()]
print logreg.predict(tfidf.transform(cv.transform(text)))