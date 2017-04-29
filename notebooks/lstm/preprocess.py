import gc
import time

import numpy as np
import pandas as pd
import regex as re
import random

import nltk
import pickle

import string
from sklearn.cross_validation import train_test_split

start0 = time.time()
interim_path = '../../interim/'

lemmatizer = nltk.stem.WordNetLemmatizer()

def process_text(text):
    tokens = []
    for word in nltk.word_tokenize(text):
        # remove punctuation, lowercase
        word = re.sub(ur"\p{P}+", "", word.lower())
        # ii. Remove all digits
        word = re.sub(ur"\d", "", word)
        # remove short words
        if len(word) < 2:
                continue 
        # lemmatize
        word = lemmatizer.lemmatize(word)
        # append the word to the full list
        tokens.append(word)

    return tokens

min_desc_length = 50
def concat(eps, train):
    shows_concat = eps[(eps['description'].str.len() > min_desc_length) & (~eps['description'].isnull())]
    print('\t%d of %d episodes were removed from %s_eps because len() < %d' % (eps.shape[0] - shows_concat.shape[0], 
                                                                               eps.shape[0], train,  min_desc_length))

    # concatenate episode descriptions by show
    shows_concat = shows_concat.groupby(['podcast_name' , 'subgenre']).apply(lambda x: ' '.join(x['description']))
    shows_concat = shows_concat.reset_index()
    shows_concat.columns = ['podcast_name', 'subgenre', 'description']

    print("\t%d unique podcasts with concatenated descriptions" % shows_concat.shape[0])
    return shows_concat

# ~~~~~~~~
# Specify whether you want the sample or not
samp = True
# ~~~~~~~~
samp = '_samp' if samp else ''

print '\nPre-pocessing', samp 

# LOAD --------------------------------------------------

# Load files
print "\nLoading data..."
start = time.time()
     
with open( interim_path + 'pods' + samp + '.p') as p:
	dfp = pickle.load(p)

podcast_array = dfp.as_matrix(['podcast_name','show_desc','subgenre'])

with open( interim_path + 'eps' + samp + '.p') as p:
	dfe = pickle.load(p)
episode_array=dfe[['podcast_name','description','subgenre']].as_matrix() 

print "\tPodcast data: ", podcast_array.shape
print "\tEpisode data: ", episode_array.shape

print "Time elapsed: {:.2} minutes".format((time.time() - start) / 60)

# CONCATENATE EPISODES BY SHOW --------------------------

print "\nConcatenating episode descriptions to show description..."
start = time.time()

for podcast in podcast_array:
    if podcast[1] is None: 
        podcast[1] = ''
    for episode in episode_array:
        if podcast[0]==episode[0] and episode[1] is not None:
            if len(episode[1])>50:
                podcast[1] += ' ' + episode[1]

                                     
print "Time elapsed: {:.2} minutes".format((time.time() - start) / 60)

# PROCESS -------------------------------------------------

print("\nProcessing...")
start = time.time()

for podcast in podcast_array:
    podcast[1] = process_text(podcast[1])
    
print "Time elapsed: {:.2} minutes".format((time.time() - start) / 60)

# split out train and dev
#train_pods, dev_pods = train_test_split(podcast_array, test_size = 0.25)

# SAVE -------------------------------------------------

print("Saving to file...")
start = time.time()

filepath = interim_path + 'preprocessed_concatenated_all' + samp + '.p'
with open( filepath, 'wb') as f:
    pickle.dump(podcast_array,f)
print "\tSaved podcasts with concatenated descriptions to:\n\t  %s" % filepath

print "Time elapsed: {:.2} minutes".format((time.time() - start) / 60)
print("\nTotal elapsed time: {:.2} minutes".format((time.time() - start0) / 60))

