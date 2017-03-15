### Podcast Micro-Categories

# Notes:
#     Before running this script, you should have the following project directory structure:
#  
#          podcast_micro_categories/
#               /raw   (this is where the zip files are stored)
#               /scripts  (this is where this script should be stored)
#               /interim (this is where the feather objects will be stored)
#
#     You should run this script from the root of the project directory structure ("podcast_micro_categories")

from __future__ import print_function
from __future__ import division

from nltk.stem import WordNetLemmatizer


####  functions below require: nltk.download('stopwords','punkt','wordnet')

# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

#tokenizer
def tokenize(words):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as its own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(words) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#stem
def stem(words):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as its own token
    tokens = [word for sent in nltk.sent_tokenize(words) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

#lemmatize
def lemmatize(words):
    lem = WordNetLemmatizer()
    lems = [lem.lemmatize(word) for word in words]
    return lems