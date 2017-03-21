### Podcast Micro-Categories
### Author: Samantha Brownstein & Jason Blauvelt

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

import gc
import os
import re
import unicodedata
import zipfile as zf
import pandas as pd
import feather

def try_norm(x):
    try:
        return unicodedata.normalize('NFKD', x)
    except TypeError:
        return ''
    
# Step 1: unzip files

ep_list = []
pod_list = []
for item in os.listdir('raw'): # loop through items in /raw
    if item.endswith('.zip'): # check for ".zip" extension

        print(item)

        zip_ref = zf.ZipFile('raw/' + item) # create Zipfile object
        
        # Extract both files directly into memory, parse as Pandas DataFrame (adding the subgenre column),
        # and add to list of DFs, which will be combined later
        for fl in zip_ref.namelist():
            if fl.startswith('ep_'):
                df = pd.read_csv(zip_ref.open(fl), encoding = "ISO-8859-1")
                df['subgenre'] = [re.sub('\\.zip', '', item)] * df.shape[0]
                ep_list.append(df)
            elif fl.startswith('pod_'):
                df = pd.read_csv(zip_ref.open(fl), encoding = "ISO-8859-1")
                df['subgenre'] = [re.sub('\\.zip', '', item)] * df.shape[0]
                pod_list.append(df)

        zip_ref.close() # close file

# Step 2: Combine lists of dataframes

# Concatenate episodes tables
eps = pd.concat(ep_list)
print(eps.shape)

# Concatenate podcast (show) tables
pods = pd.concat(pod_list)
print(pods.shape)
        
del ep_list, pod_list
gc.collect()

# Step 3: Convert unicode to ASCII
def try_norm(x):
    try:
        return unicodedata.normalize('NFKD', x).encode('ascii', errors='ignore')
    except TypeError:
        return None
    
eps['description'] = eps['description'].map(try_norm)
pods['show_desc'] = pods['show_desc'].map(try_norm)


# Step 4: Save dataframes as feather objects
feather.write_dataframe(eps,'interim/eps.feather') 
feather.write_dataframe(pods,'interim/pods.feather')

# Save 10% samples too
pods_samp = pods.sample(frac=.1)
eps_samp = eps[eps['podcast_name'].isin(pods_samp['podcast_name'])]

feather.write_dataframe(eps_samp, 'interim/eps_samp.feather')
feather.write_dataframe(pods_samp, 'interim/pods_samp.feather')

