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

from sklearn.feature_extraction.text import CountVectorizer
import string
import regex as re
import nltk
from nltk.corpus import stopwords



### I. Define helper sets -------------------------------------------

# A. Stop words
stop_word_set = set(stopwords.words("english"))

# B. Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

        
### II. Tokenizer function -------------------------------------------------------

# Tokenize a single text block
def tokenize(text, rmv_all_digits = False, require_letter = False, 
             canonicalize_digits = False, lowercase = False, lemmatizer = None,
             stemmer = None, canonicalize_word = False, canon = None):

    # For lemmatizer: Try nltk.stem.WordNetLemmatizer()
    # For stemmer: Try nltk.stem.SnowballStemmer('english')
    
    # 1. Check argument consistency
    if canonicalize_word and not canon:
        return "Error: If canonicalize_word = True, you need to specify *canon*"
    if canonicalize_word:
        print("NOTE: Make sure your canon is preprocessed in the same way as the arguments you specify. So if you want to lemmatize, lemmatize your canon too.")
    if lemmatizer and stemmer:
        return "Do not lemmatize *and* stem - choose one"
    
   
    # 2. Loop through the words in the text block and apply preprocessing steps
    tokens = []
    for word in nltk.word_tokenize(text):

        # i. Remove punctuation (always on)
        word = re.sub(ur"\p{P}+", "", word)
        # ii. Remove all digits
        if rmv_all_digits:
            word = re.sub(ur"\d", "", word)
        # iii. Check word length
        if len(word) < 2:
                continue
        # iv. Require letter
        if require_letter and not re.search('[a-zA-Z]', word):
            continue
        # v. lemmatize
        if lemmatizer:
            word = lemmatizer.lemmatize(word)
        # vi. stem
        if stemmer:
            word = stemmer.stem(word)
        # vii. Canonicalize digits (convert digits to *uppercase* DG)
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


# III. Episode concatenate function ------------------------------------------------------------

def concat_eps_by_pod(df, min_desc_length = 50):

    # 1. Filter to episodes longer than the desired length
    comb = df[(df['description'].str.len() > min_desc_length) & (~df['description'].isnull())]
    print('%d / %d episodes removed because len() < %d' % (df.shape[0] - comb.shape[0], df.shape[0], min_desc_length))

    # 2. Concatenate episode descriptions by show
    comb = comb.groupby(['podcast_name' , 'subgenre']).apply(lambda x: ' '.join(x['description']))
    comb = comb.reset_index()
    comb.columns = ['podcast_name', 'subgenre', 'description']
    print("%d unique podcasts with concatenated descriptions" % comb.shape[0])

    return comb
