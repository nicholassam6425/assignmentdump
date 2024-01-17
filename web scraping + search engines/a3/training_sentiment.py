import argparse
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from joblib import dump

#arg parse stuff
parser = argparse.ArgumentParser(
    prog='train sentiment',
    description='trains sentiment analysis model'
)
parser.add_argument('--imdb', action='store_true', help="use imdb data")
parser.add_argument('--amazon', action='store_true', help="use amazon data")
parser.add_argument('--yelp', action='store_true', help="use yelp data")
parser.add_argument('--naive', action='store_true', help="use naive bayes")
parser.add_argument('--knn', type=int, help="use knn model")
parser.add_argument('--svm', action='store_true', help="use svm model")
parser.add_argument('--decisiontree', action='store_true', help="use decision tree")
args = parser.parse_args()
#x = features, y = target
x = []
y = []

#open files depending on args, also include default
if args.imdb:
    imdb = open("imdb_labelled.txt", "r")
if args.amazon:
    amazon = open("amazon_cells_labelled.txt", "r")
if args.yelp:
    yelp = open("yelp_labelled.txt", "r")
if not args.imdb and not args.amazon and not args.yelp:
    print("No data was chosen, defaulting to IMDB.")
    imdb = open("imdb_labelled.txt", "r")
    for i in imdb.readlines():
        x.append(i.split("\t")[0].strip())
        y.append(int(i.split("\t")[1].strip()))

#read data from files
if args.imdb:
    for i in imdb.readlines():
        x.append(i.split("\t")[0].strip())
        y.append(int(i.split("\t")[1].strip()))
if args.amazon:
    for i in amazon.readlines():
        x.append(i.split("\t")[0].strip())
        y.append(int(i.split("\t")[1].strip()))
if args.yelp:
    for i in yelp.readlines():
        x.append(i.split("\t")[0].strip())
        y.append(int(i.split("\t")[1].strip()))

#transform data to features readable by model (tfidf document)
count_vect = CountVectorizer(stop_words='english', strip_accents='ascii')
X_train_counts = count_vect.fit_transform(x)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#create model based on input, with default
if args.naive:
    clf = GaussianNB()
elif args.knn:
    clf = KNeighborsClassifier(n_neighbors=args.knn)
elif args.svm:
    clf = SVC()
elif args.decisiontree:
    clf = DecisionTreeClassifier()
else:
    print("No model was chosen. Defaulting to Naive Bayes.")
    clf = GaussianNB()
clf.fit(X_train_tfidf.toarray(), y)
#use cross_validation to measure scores per n_slice
scores = cross_validate(clf, X_train_tfidf.toarray(), y, scoring=['accuracy', 'recall', 'precision', 'f1'])

#get rid of unnecessary data then print
del scores['fit_time']
del scores['score_time']
print(scores)

#dump model, countvectorizer, and tfidf_transformer for predict_sentiment to use
dump(clf, 'model.dump')
dump(count_vect, 'cv.dump')
dump(tfidf_transformer, 'tfidf.dump')