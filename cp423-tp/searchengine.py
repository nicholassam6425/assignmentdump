from sklearn.calibration import CalibratedClassifierCV
from joblib import load
from pathlib import Path
import warnings
from joblib import dump
import glob
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import ast
from os import path
import hashlib
import os
import string
from bs4 import BeautifulSoup
from dateutil.parser import parse
from justext import justext
import nltk
import requests
import ssl
import string
nltk.download('stopwords')
warnings.filterwarnings("ignore")


def extract_content_and_links(url):
    ssl._create_default_https_context = ssl._create_unverified_context
    stopwords = nltk.corpus.stopwords.words('english')
    stoplist = tuple(nltk.corpus.stopwords.words('english'))

    # Get the HTML content of the page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html5lib')

    date_element = soup.find('meta', {'property': 'article:published_time'})
    if date_element is not None:
        date_str = date_element['content']
        date = parse(date_str)
    else:
        date = None

    # Extract the text content of the page using justext
    paragraphs = justext(response.content, stoplist=stoplist)
    content = '\n'.join([p.text for p in paragraphs if not p.is_boilerplate])

    # Extract links from the page
    links = [a['href'] for a in soup.find_all('a') if 'href' in a.attrs]

    return content, links, date


def train_classifier():
    if len(glob.glob('data/*/*.txt')) == 0:
        print("Please 'Collect New Documents (1)' first")
        return
    y = []
    x = []
    for filename in glob.glob('data/*/*.txt'):
        y.append(os.path.basename(os.path.dirname(filename)))
        with open(filename, 'r', encoding='utf-8') as file:
            x.append(file.read())
    count_vect = CountVectorizer(stop_words='english', strip_accents='ascii')
    X_train_counts = count_vect.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    sgd = SGDClassifier()

    sgd.fit(X_train_tfidf.toarray(), y)
    calibrator = CalibratedClassifierCV(sgd, cv='prefit')
    proba_model = calibrator.fit(X_train_tfidf.toarray(), y)

    sgdscores = cross_validate(sgd, X_train_tfidf.toarray(), y, scoring=[
                               'accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted'])

    del sgdscores['fit_time']
    del sgdscores['score_time']

    y_pred = sgd.predict(X_train_tfidf.toarray())
    print("confusion matrix: ")
    print(confusion_matrix(y, y_pred))
    print("performance metrics: ")
    print(sgdscores)
    dump(proba_model, 'proba.dump')
    dump(sgd, 'classifier.dump')
    dump(count_vect, 'cv.dump')
    dump(tfidf_transformer, 'tfidf.dump')


def predict_link(url):
    model_check = Path('classifier.dump')
    cv_check = Path('cv.dump')
    tfidf_check = Path('tfidf.dump')
    proba_check = Path('proba.dump')
    if not (model_check.is_file() and cv_check.is_file() and tfidf_check.is_file() and proba_check.is_file()):
        print("Please 'Train ML Classifier (4)' first.")
        return
    stoplist = tuple(nltk.corpus.stopwords.words('english'))
    response = requests.get(url)
    paragraphs = justext(response.content, stoplist=stoplist)
    content = ['\n'.join([p.text for p in paragraphs if not p.is_boilerplate])]
    clf = load('classifier.dump')
    count_vect = load('cv.dump')
    X_new_counts = count_vect.transform(content)
    tfidf_transformer = load('tfidf.dump')
    X_tfidf = tfidf_transformer.transform(X_new_counts)
    proba_model = load('proba.dump')
    pred = clf.predict(X_tfidf.toarray())
    pred_proba = proba_model.predict_proba(X_tfidf.toarray())
    if pred[0] == 'Cybersecurity':
        print(f'<{pred[0]}, {pred_proba[0][0]*100:.2f}%>')
    if pred[0] == 'Health':
        print(f'<{pred[0]}, {pred_proba[0][1]*100:.2f}%>')
    if pred[0] == 'Technology':
        print(f'<{pred[0]}, {pred_proba[0][2]*100:.2f}%>')


def remove_stopwords(content, stopwords):

    tokens = nltk.word_tokenize(content)
    filtered_tokens = [token.lower()
                       for token in tokens if token.lower() not in stopwords]
    filtered_content = ' '.join(filtered_tokens)

    return filtered_content


def save_page_content(topic, url, content, date, mapping):
    # Calculate the hash value of the URL
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()

    # Get the current DocID based on the number of records in the mapping file
    doc_id = len(mapping)

    # Add a new mapping record
    mapping[url_hash] = doc_id

    # Create the topic related subfolder
    subfolder = 'data/' + topic

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    stopwords = nltk.corpus.stopwords.words('english')
    content = remove_stopwords(content, stopwords)

    # Save the page content
    try:
        with open(os.path.join(subfolder, f'{doc_id}.txt'), 'w', encoding='utf-8') as file:
            file.write(content)
    except UnicodeEncodeError:
        with open(os.path.join(subfolder, f'{doc_id}.txt'), 'w', encoding='utf-8', errors='ignore') as file:
            file.write(content)

    with open('crawl.log', 'a') as file:
        file.write(f'{topic}, {url}, H{doc_id}, {date}\n')


def process_sources_file(file_path):
    with open(file_path, 'r') as file:
        sources = file.readlines()

    mapping = {}

    if os.path.exists('mapping.txt'):
        with open('mapping.txt', 'rb') as file:
            # mapping = pickle.load(file)
            None
    if os.path.exists('crawl.log'):
        open('crawl.log', 'w').close()

    for source in sources:
        topic, url = source.split(', ')
        topic = topic.strip()
        url = url.strip()

        content, links, date = extract_content_and_links(url)
        filtered_content = remove_stopwords(
            content, nltk.corpus.stopwords.words('english'))
        save_page_content(topic, url, filtered_content, date, mapping=mapping)

        for link in links:
            if link.startswith(url):
                content, links, date = extract_content_and_links(link)
                filtered_content = remove_stopwords(
                    content, nltk.corpus.stopwords.words('english'))
                save_page_content(
                    topic, link, filtered_content, date, mapping=mapping)

    with open('mapping.txt', 'wb') as file:
        # pickle.dump(mapping, file)
        None


def soundex_generator(token):
    # Convert the word to upper
    # case for uniformity
    token = token.upper()
    soundex = ""
    # Retain the First Letter
    soundex += token[0]
    # Create a dictionary which maps
    # letters to respective soundex
    # codes. Vowels and 'H', 'W' and
    # 'Y' will be represented by '.'
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2",
                  "DT": "3",
                  "L": "4", "MN": "5", "R": "6",
                  "AEIOUHWY": "."}
    # Enode as per the dictionary
    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code

    # Trim or Pad to make Soundex a
    # 4-character code
    soundex = soundex[:4].ljust(4, "0")

    return soundex

# Define a function to tokenize a document


def tokenize_document(document):
    # Remove punctuation from the document and convert it to lowercase
    document = document.translate(
        str.maketrans('', '', string.punctuation)).lower()
    # Tokenize the document using NLTK
    tokens = nltk.word_tokenize(document)
    # Remove stop words from the token list using NLTK
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords]
    # Return the token list
    return tokens

# Define a function to create the inverted index


def create_inverted_index():
    nltk.download('punkt')
    # Create an empty dictionary to hold the inverted index
    inverted_index = {}
    # Loop over all the files in the data directory
    id = 1
    if not os.path.exists('mapping.txt'):
        open('mapping.txt', 'w+').close()

    with open('mapping.txt', 'r+') as map:
        for subdir, _, files in os.walk('data'):
            for file in files:
                # Read the contents of the file
                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                # Tokenize the contents of the file
                tokens = tokenize_document(content)
                # Calculate the hash of the file path
                document_hash = hashlib.sha256(os.path.join(
                    subdir, file).encode('utf-8')).hexdigest()

                # find the docID for the document_hash using mapping.txt
                with open('mapping.txt', 'a') as map:
                    map.write(f'{document_hash} | H{id}\n')

                # change variable document_hash to docID beyond this point
                docID = 'H' + str(id)
                id += 1

                # Update the inverted index with the tokens
                for token in tokens:
                    if token not in inverted_index:
                        inverted_index[token] = set()
                    inverted_index[token].add((docID, tokens.count(token)))
        # Invert the inverted index to get the final inverted index
        final_inverted_index = {}
        for term, document_set in inverted_index.items():
            final_inverted_index[term] = []
            for document in document_set:
                final_inverted_index[term].append((document[0], document[1]))
        # Write the inverted index to a file
        with open('invertedindex.txt', 'w', encoding='utf-8') as f:
            f.write('| Term | Soundex | Appearances (DocHash, Frequency) |\n')
            f.write('|------|---------|----------------------------------|\n')
            for term, document_list in sorted(final_inverted_index.items()):
                f.write(
                    f'| {term} | {soundex_generator(term)} | {document_list} |\n')


def query_search(query):
    words = query.split(' ')
    with open('invertedindex.txt', 'r', encoding='utf-8') as ii:
        rawterms = ii.readlines()
        rawterms.remove(rawterms[0])
        rawterms.remove(rawterms[0])
        docs = {}
        for term in rawterms:
            temp = term.split('|')
            docs[temp[1].strip()] = {
                'soundex': temp[2].strip(), 'docs': temp[3].strip()}
        new_words = []
        for word in words:
            if word not in docs.keys():
                soundex = soundex_generator(word)
                for k, v in docs.items():
                    if soundex == v['soundex']:
                        new_words.append(k)
                        break
            else:
                new_words.append(word)
        match_docs = []
        for word in new_words:
            for k, v in docs.items():
                if word == k:
                    for i in ast.literal_eval(v['docs']):
                        match_docs.append(i)
        match_docs.sort(key=lambda x: -x[1])
        doc_links = []
        with open('crawl.log', 'r', encoding='utf-8') as rawlinks:
            rawlinklist = rawlinks.readlines()
            linklist = {}
            for link in rawlinklist:
                temp = link.split(',')
                linklist[temp[2].strip()] = temp[1].strip()
            c = 0
            for doc in match_docs:
                if doc[0] in linklist.keys():
                    doc_links.append(linklist[doc[0]])
                    c += 1
                if c >= 3:
                    break
        print('Found the following hashed link(s):')
        for i in doc_links:
            print(i)


def print_story():
    f = open('story.txt', 'r')

    lines = f.readlines()

    for line in lines:
        print(line.strip())

    f.close()


def print_options():
    print('\n\nSelect an option:\n')
    print('1-Collect new documents')
    print('2-Index documents')
    print('3-Search for a query')
    print('4-Train ML classifier')
    print('5-Predict a link')
    print('6-Your story!')
    print('7-Exit')


while True:
    print_options()
    ch = int(input('\n'))
    if ch == 1:
        process_sources_file('sources.txt')
    elif ch == 2:
        create_inverted_index()
    elif ch == 3:
        query_search(input("\nEnter your query: "))
    elif ch == 4:
        train_classifier()
    elif ch == 5:
        predict_link(input("\nEnter a link: "))
    elif ch == 6:
        print_story()
    elif ch == 7:
        break
    else:
        print("Please enter a valid option.")
        print_options()
