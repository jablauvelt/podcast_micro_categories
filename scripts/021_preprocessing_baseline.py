### Podcast Micro-Categories
### Preprocessing Baseline
### Description:
###   This script performs only the most basic preprocessing
###   steps on the podcast data, to be used for the baseline.

### SETUP: This script should be called from the project root
###    directory (/podcast_micro_categories/)

from __future__ import print_function
from __future__ import division

import gc

import numpy as np
import pandas as pd
import pickle
import feather

import module_preprocess


# I. LOAD --------------------------------------------------

# Load sample files
with open('interim/pods_samp.p') as p:
	pods_df_samp = pickle.load(p)
with open('interim/eps_samp.p') as p:
	eps_df_samp = pickle.load(p)

# Load full files
with open('interim/pods.p') as p:
	pods_df = pickle.load(p)
with open('interim/eps.p') as p:
	eps_df = pickle.load(p)
                                     
print("Episodes table shape (sample): ", eps_df_samp.shape)
print("Podcasts table shape (sample): ", pods_df_samp.shape)
print("Episodes table shape: ", eps_df.shape)
print("Podcasts table shape: ", pods_df.shape)

# II. TOKENIZE ---------------------------------------------

# This step only tokenizes and removes punctuation

# Samples
print("Processing samples")
pods_df_samp['show_desc'] = pods_df_samp['show_desc'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')
eps_df_samp['description'] = eps_df_samp['description'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')

# Full sets
print("Processing full sets")
pods_df['show_desc'] = pods_df['show_desc'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')
eps_df['description'] = eps_df['description'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')

# III. EXPORT -------------------------------------------------

# Samples
print("Exporting samples")
with open('interim/021_preproc_baseline_pods_samp.p', 'wb') as fp:
    pickle.dump(pods_df_samp, fp)
with open('interim/021_preproc_baseline_eps_samp.p', 'wb') as fp:
    pickle.dump(eps_df_samp, fp)

del pods_df_samp
del eps_df_samp
gc.collect()
    
# Full sets
print("Exporting full sets")
with open('interim/021_preproc_baseline_pods.p', 'wb') as fp:
    pickle.dump(pods_df, fp)
del pods_df
gc.collect()
with open('interim/021_preproc_baseline_eps.p', 'wb') as fp:
    pickle.dump(eps_df, fp)
