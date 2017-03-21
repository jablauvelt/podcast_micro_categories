### Podcast Micro-Categories
### Pre-processing Functions
### Description:
###   This script only defines preprocessing objects and functions.
###   It is to be called on by other scripts as needed.

# NOTE: Before running for the first time, run the following commands: 
#        nltk.download('stopwords')
#        nltk.download('wordnet') 
#        nltk.download('punkt') 

from __future__ import print_function
from __future__ import division

import string
import re
import nltk
from nltk.corpus import stopwords


### I. Define helper sets -------------------------------------------

# A. Stop words
stop_word_set = set(stopwords.words("english"))

        
### II. Tokenizer function -------------------------------------------------------

# Tokenize a single text block
def tokenize(text, rmv_all_digits = False, require_letter = False, 
             canonicalize_digits = False, rmv_stopwords = False, stop_word_set = None, 
             lowercase = False, lemmatize = False, lemmatizer = None,
             stem = False, stemmer = None, canonicalize_word = False, canon = None):
    
    # Check argument consistency
    if rmv_stopwords and not stop_word_set:
        return "Error: If rmv_stopwords = True, you need to specify *stop_word_set*"
    if canonicalize_word and not canon:
        return "Error: If canonicalize_word = True, you need to specify *canon*"
    if canonicalize_word:
        print("NOTE: Make sure your canon is preprocessed in the same way as the arguments you specify. So if you want to lemmatize, lemmatize your canon too.")
    if lemmatize and stem:
        return "Do not lemmatize *and* stem - choose one"
    if lemmatize and not lemmatizer:
        return "Error: If lemmatizer = True, you need to specify *lemmatizer*. Try nltk.stem.WordNetLemmatizer()."
    if stem and not stemmer:
        return "Error: If stemmer = True, you need to specify *stemmer*. Try nltk.stem.SnowballStemmer('english')."
    
   
    # Loop through the words in the text block and apply preprocessing steps
    tokens = []
    for word in nltk.word_tokenize(text):
        # i. Remove punctuation (always on)
        word = word.translate(None, string.punctuation)
        if not word:
            continue
        # ii. Remove all digits
        if rmv_all_digits:
            word = word.translate(None, string.digits)
            if not word:
                continue
        # iii. Require letter
        if require_letter and not re.search('[a-zA-Z]', word):
            continue
        # iv. Remove stopwords 
        if rmv_stopwords and word.lower() in stop_word_set:
            continue
        # v. Lowercase 
        if lowercase:
            word = word.lower()
        # vi. lemmatize
        if lemmatize:
            word = lemmatizer.lemmatize(word)
        # vii. stem
        if stem:
            word = stemmer.stem(word)
        # viii. Canonicalize digits (convert digits to *uppercase* DG)
        if canonicalize_digits: 
            word = re.sub("\d", "DG", word)
        # viii. Canonicalize word. If you want to include words with
        #     digits, make sure some version of DG / DGDG, etc. is in
        #     your wordset. If you also specified lowercase, lemmatize, or stem,
        #     make sure you lowercased/lemmatized/stemmed your canon respectively
        if canonicalize_word and word not in canon:
            word = '<unk>'
            
        # Now that all the preprocessing steps have been applied to this word,
        # append the word to the full list
        tokens.append(word)
    
    return tokens
    