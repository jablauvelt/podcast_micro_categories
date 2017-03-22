### Podcast Micro-Categories
### Preprocessing Heavy
### Description:
###   This script performs "heavy" preprocessing, aka all of the
###   most restrictive preprocessing steps.

### SETUP: This script should be called from the project root
###    directory (/podcast_micro_categories/)

from __future__ import print_function
from __future__ import division

import gc
import time

import numpy as np
import pandas as pd
import nltk
import pickle

import module_preprocess

start0 = time.time()

# I. LOAD --------------------------------------------------

# Load files
start = time.time()
with open('interim/pods_samp.p') as p:
	pods = pickle.load(p)
with open('interim/eps_samp.p') as p:
	eps = pickle.load(p)
                                     
print("Episodes table shape: ", eps.shape)
print("Podcasts table shape: ", pods.shape)
print((time.time() - start) / 60)

# II. CONCATENATE EPISODES BY SHOW----------------------------

print("Concatenating episodes by show")
start = time.time()
eps = module_preprocess.concat_eps_by_pod(eps)

print((time.time() - start) / 60)

# III. TOKENIZE ---------------------------------------------

# Define lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Define tokenizer
def tokenizer(x):
	return module_preprocess.tokenize(x,  rmv_all_digits = True, rmv_stopwords = True, stop_word_set = module_preprocess.stop_word_set, 
             						   lowercase = True, lemmatize = True, lemmatizer = lemmatizer)


print("Processing")
start = time.time()
pods['show_desc_tokens'] = pods['show_desc'].map(lambda x: tokenizer(x), na_action = 'ignore')
eps['description_tokens'] = eps['description'].map(lambda x: tokenizer(x), na_action = 'ignore')

print((time.time() - start) / 60)

# IV. EXPORT -------------------------------------------------

print("Exporting")
start = time.time()
pods.to_pickle('interim/028_preproc_heavy_pods.p')
eps.to_pickle('interim/028_preproc_heavy_eps.p')

print((time.time() - start) / 60)

print((time.time() - start0) / 60)