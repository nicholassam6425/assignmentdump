import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import ijson
import string

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

def generate_candidates(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
    replaces = [left + char + right[1:] for left, right in splits if right for char in alphabet]
    inserts = [left + char + right for left, right in splits for char in alphabet]
    candidates = list(set(deletes + transposes + replaces + inserts))
    return candidates

parser = argparse.ArgumentParser(
    prog='NoisyChannel',
    description='Noisy Channel Model for spellchecking unigrams'
)
parser.add_argument('--correct', action='store_true', help="Output likeliest correct words")
parser.add_argument('--proba', action='store_true', help="Outputs probability of each word being correct")
parser.add_argument('words', metavar='words', type=str, nargs='+', help='List of words to check')
args = parser.parse_args()
if args.correct and args.proba:
    print("--proba and --correct are mutually exclusive. Call them one at a time.")
    exit()
words = {}
word_freq = {}
data = load_wikipedia()
total_words = 0
for article in data:
    text = article['text']
    text = preprocess(text)
    tokens = word_tokenize(text)
    for token in tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1
        total_words += 1

with open('wikipedia.corpus', 'w', encoding='utf-8') as f:
    for pair in word_freq:
        f.write(f'{pair}: {word_freq[pair]}\n')


with open('wikipedia.corpus', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        word = line.split(": ")[0]
        num = line.split(": ")[1]
        words[word] = num
#writes to file because initially, we used wikipedia_processing.py to write wikipedia.corpus
#i don't really want to think about how to change it to work without file, so it writes to file.

words_in = {}
for i in args.words:
    for j in i.strip('][').split(','):
        words_in[j] = generate_candidates(j)
output = {}
for key in words_in:
    if args.proba:
        output[key] = []
    if args.correct:
        best_word = words_in[key][0]
    
    for candidate in words_in[key]:
        if candidate in words:
            if args.correct:
                if best_word in words:
                    if words[best_word] < words[candidate]:
                        best_word = candidate
                else:
                    best_word = candidate
            if args.proba:
                output[key].append(f'{candidate}: {int(words[candidate])/total_words}')
    if args.correct:
        output[key] = best_word
print(output)
