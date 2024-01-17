from datetime import datetime
from os import path
import hashlib
import os

from bs4 import BeautifulSoup
from dateutil.parser import parse
from justext import justext
import nltk
import requests
import ssl
nltk.download('stopwords')

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
import glob
from joblib import dump

if len(glob.glob('data/*/*.txt')) == 0:
    print("Please 'Collect New Documents (1)' first")
y = []
x = []
for filename in glob.glob('data/*/*.txt'):
    y.append(filename.split("\\")[1])
    with open(filename, 'r', encoding='utf-8') as file:
        x.append(file.read())
count_vect = CountVectorizer(stop_words='english', strip_accents='ascii')
X_train_counts = count_vect.fit_transform(x)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

sgd = SGDClassifier()

sgd.fit(X_train_tfidf.toarray(), y)

sgdscores = cross_validate(sgd, X_train_tfidf.toarray(), y, scoring=['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted'])

del sgdscores['fit_time']
del sgdscores['score_time']

y_pred = sgd.predict(X_train_tfidf.toarray())
print("confusion matrix: ")
print(confusion_matrix(y, y_pred))
print("performance metrics: ")
print(sgdscores)

dump(sgd, 'classifier.dump')
dump(count_vect, 'cv.dump')
dump(tfidf_transformer, 'tfidf.dump')