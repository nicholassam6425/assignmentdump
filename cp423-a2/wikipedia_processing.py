
import argparse
import json
import os
import ssl
import string
from collections import defaultdict
import ijson

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#import pdb
#pdb.set_trace()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('stopwords')


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zipf', action='store_true', help='Plot Zipf diagram')
    parser.add_argument('--tokenize', action='store_true', help='Tokenize corpus')
    parser.add_argument('--stopword', action='store_true', help='Remove stopwords from corpus')
    parser.add_argument('--stemming', action='store_true', help='Apply stemming to corpus')
    parser.add_argument('--invertedindex', action='store_true', help='Create inverted index of corpus')
    return parser.parse_args()

def load_wikipedia():
    data = []
    for filename in os.listdir('data_wikipedia'):
        if filename.endswith('.json'):
            with open(os.path.join('data_wikipedia', filename), 'rb') as f:
                parser = ijson.parse(f)
                article = {}
                try:
                    for prefix, event, value in parser:
                        if prefix == 'item.text':
                            article['text'] = value
                except Exception as e:
                    print(f"Error occurred while reading {filename}: {e}")
                data.append(article)
    return data

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(data):
    word_freq = {}
    for article in data:
        text = article['text']
        text = preprocess(text)
        tokens = word_tokenize(text)
        for token in tokens:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1

    with open('wikipedia.corpus', 'w', encoding='utf-8') as f:
        for pair in word_freq:
            f.write(f'{pair}: {word_freq[pair]}\n')

def remove_stopwords(tokens):
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens

def apply_stemming(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_tokens.append(stemmed_token)
    return stemmed_tokens

def plot_zipf(tokens):
    fdist = FreqDist(tokens)
    fdist.plot(30, cumulative=False)
    plt.show()


def create_inverted_index(data):
    word_counts = defaultdict(lambda: defaultdict(int))
    for i, article in enumerate(data):
        text = article['text']
        text = preprocess(text)
        tokens = word_tokenize(text)
        for token in set(tokens):
            count = tokens.count(token)
            word_counts[token][i] = count

    # sort the words by frequency in descending order
    sorted_words = sorted(word_counts.keys(), key=lambda w: sum(word_counts[w].values()), reverse=True)

    # create the inverted index dictionary
    inverted_index = defaultdict(list)
    for rank, word in enumerate(sorted_words):
        for file_num, count in word_counts[word].items():
            inverted_index[word].append((file_num, count))

    return inverted_index


def write_output(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in data:
            f.write(token + '\n')

def main():
    args = get_args()
    data = load_wikipedia()
    #print(args)

    if args.tokenize:
        #print(args.tokenize)
        print("in tokenization")
        tokens = tokenize(data)
        write_output('data_wikipedia/wikipedia.token', tokens)
        print('Tokenization done.')
    if args.stopword:
        tokens = tokenize(data)
        filtered_tokens = remove_stopwords(tokens)
        write_output('data_wikipedia/wikipedia.token.stop', filtered_tokens)
        print('Stopword removal done.')
    if args.stemming:
        tokens = tokenize(data)
        stemmed_tokens = apply_stemming(tokens)
        write_output('data_wikipedia/wikipedia.token.stem', stemmed_tokens)
        print('Stemming done.')
    if args.zipf:
        tokens = tokenize(data)
        plot_zipf(tokens)
        
    if args.invertedindex:
        inverted_index = create_inverted_index(data)
        with open('data_wikipedia/wikipedia.invertedindex', 'w') as f:
            json.dump(inverted_index, f)
        print('Inverted index created.')

if __name__ == '__main__':
    main()