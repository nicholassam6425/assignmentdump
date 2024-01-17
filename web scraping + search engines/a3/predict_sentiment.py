import sys
from joblib import load
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

model_check = Path('model.dump')
cv_check = Path('cv.dump')
tfidf_check = Path('tfidf.dump')
if not (model_check.is_file() and cv_check.is_file() and tfidf_check.is_file()):
    print("One of the required files from training_sentiment.py is missing. Try running training_sentiment.py again.")
    exit()

clf = load('model.dump')
count_vect = load('cv.dump')
X_new_counts = count_vect.transform(sys.argv[1:])
tfidf_transformer = load('tfidf.dump')
X_tfidf = tfidf_transformer.transform(X_new_counts)
pred = clf.predict(X_tfidf.toarray())
for i in range(0, len(pred)):
    if pred[i] == 1:
        print(f"The text \"{sys.argv[i+1]}\" is positive")
    elif pred[i] == 0:
        print(f"The text \"{sys.argv[i+1]}\" is negative")
    else:
        print("Error. Predicted neither positive or negative.")