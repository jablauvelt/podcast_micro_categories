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
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pickle

import module_preprocess

start0 = time.time()

# ~~~~~~~~
# Specify whether you want the sample or not
samp = False
# ~~~~~~~~
samp = '_samp' if samp else ''

# I. LOAD --------------------------------------------------

# Load files
start = time.time()
with open('interim/pods' + samp + '.p') as p:
	pods = pickle.load(p)
with open('interim/eps' + samp + '.p') as p:
	eps = pickle.load(p)
                                     
print("Episodes table shape: ", eps.shape)
print("Podcasts table shape: ", pods.shape)
print("Loading took: {:.2} minutes".format((time.time() - start) / 60))

# II. CONCATENATE EPISODES BY SHOW----------------------------

print("Concatenating episodes by show")
start = time.time()
shows_concat = module_preprocess.concat_eps_by_pod(eps)

print("Concatenation took: {:.2} minutes".format((time.time() - start) / 60))

# III. CONVERT TO TERM DOCUMENT MATRIX ---------------------------------------------

print("Processing")
start = time.time()

# Create a Term Document Matrix out of the descriptions
vectorizer = CountVectorizer(stop_words='english', min_df = 10, max_df=.1, 
                             tokenizer= lambda x: module_preprocess.tokenize(x, rmv_all_digits=True, 
                                                           lemmatizer=module_preprocess.lemmatizer))
tdm = vectorizer.fit_transform(shows_concat['description'])

print("Processing took: {:.2} minutes".format((time.time() - start) / 60))

# IV. EXPORT -------------------------------------------------

print("Exporting")
start = time.time()

# Save episode names and subgenres
shows_concat[['podcast_name', 'subgenre']].to_pickle('interim/028_preproc_heavy_shows_concat' + samp + '.p')

# Save TDM
np.savez('interim/028_preproc_heavy_tdm' + samp + '.npz', data=tdm.data, indices=tdm.indices,
         indptr=tdm.indptr, shape=tdm.shape)

# Save feature names (columns of the sparse matrix)
pd.DataFrame(vectorizer.get_feature_names(), columns=['word']).to_pickle('interim/028_preproc_heavy_names' + samp + '.p')

print("Saving took: {:.2} minutes".format((time.time() - start) / 60))
print("The whole process took: {:.2} minutes".format((time.time() - start0) / 60))

# TODO
# remove shows with fewer than X episodes
# add show summary descriptions?
