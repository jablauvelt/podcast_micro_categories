### Podcast Micro-Categories
### Preprocessing Baseline
### Description:
###   This script performs only the most basic preprocessing
###   steps on the podcast data, to be used for the baseline.

### SETUP: This script should be called from the project root
###    directory (/podcast_micro_categories/)

from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import pickle
import feather

import module_preprocess


# I. LOAD --------------------------------------------------

eps_path = 'interim/eps_samp.feather' 
pods_path = 'interim/pods_samp.feather'

eps_df = feather.read_dataframe(eps_path)
pods_df = feather.read_dataframe(pods_path)

print("Episodes table shape: ", eps_df.shape)
print("Podcasts table shape: ", pods_df.shape)

# II. TOKENIZE ---------------------------------------------

# This step only tokenizes and removes punctuation

# Samples
pods_df_samp['show_desc'] = pods_df_samp['show_desc'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')
eps_df_samp['description'] = eps_df_samp['description'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')

# Full sets
pods_df['show_desc'] = pods_df['show_desc'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')
eps_df['description'] = eps_df['description'].map(lambda x: module_preprocess.tokenize(x), na_action = 'ignore')

# III. EXPORT -------------------------------------------------

# Samples
with open('../interim/021_preproc_baseline_pods_samp.feather', 'wb') as fp:
    pickle.dump(pods_df_samp, fp)
with open('../interim/021_preproc_baseline_eps_samp.feather', 'wb') as fp:
    pickle.dump(eps_df_samp, fp)

# Full sets
with open('../interim/021_preproc_baseline_eps.feather', 'wb') as fp:
    pickle.dump(eps_df, fp)
with open('../interim/021_preproc_baseline_pods.feather', 'wb') as fp:
    pickle.dump(pods_df, fp)