from sklearn.model_selection import train_test_split
import pandas as pd
from ktext.preprocess import processor
import dill as dpickle
import numpy as np
import logging

pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#read in data sample 2M rows (for speed of tutorial)
traindf, testdf = train_test_split(pd.read_csv('data/github_issues.csv').sample(n=2000000),
                                   test_size=.10)

train_body_raw = traindf.body.tolist()
train_title_raw = traindf.issue_title.tolist()
#preview output of first element
print(train_body_raw[0])

body_pp = processor(keep_n=8000, padding_maxlen=70)
train_body_vecs = body_pp.fit_transform(train_body_raw)

print('\noriginal string:\n', train_body_raw[0], '\n')
print('after pre-processing:\n', train_body_vecs[0], '\n')

# Instantiate a text processor for the titles, with some different parameters
#  append_indicators = True appends the tokens '_start_' and '_end_' to each
#                      document
#  padding = 'post' means that zero padding is appended to the end of the
#             of the document (as opposed to the default which is 'pre')
title_pp = processor(append_indicators=True, keep_n=4500,
                     padding_maxlen=12, padding ='post')

# process the title data
train_title_vecs = title_pp.fit_transform(train_title_raw)

print('\noriginal string:\n', train_title_raw[0])
print('after pre-processing:\n', train_title_vecs[0])

# Save the preprocessor
with open('data/results/body_pp.dpkl', 'wb') as f:
    dpickle.dump(body_pp, f)

with open('data/results/title_pp.dpkl', 'wb') as f:
    dpickle.dump(title_pp, f)

# Save the processed data
np.save('data/results/train_title_vecs.npy', train_title_vecs)
np.save('data/results/train_body_vecs.npy', train_body_vecs)