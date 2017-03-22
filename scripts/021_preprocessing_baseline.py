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

# Load files
with open('interim/pods.p') as p:
	pods = pickle.load(p)
with open('interim/eps.p') as p:
	eps = pickle.load(p)
                                     
print("Episodes table shape: ", eps.shape)
print("Podcasts table shape: ", pods.shape)

# II. TOKENIZE ---------------------------------------------

# This step only tokenizes and removes punctuation

print("Processing")
pods['show_desc'] = pods['show_desc'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')
eps['description'] = eps['description'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')

# III. EXPORT -------------------------------------------------

print("Exporting full sets")
pods.to_pickle('interim/021_preproc_baseline_pods.p')
eps.to_pickle('interim/021_preproc_baseline_eps.p')
