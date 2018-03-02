#!/usr/bin/env python

# BOW + linear regression

import os
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from utility import hatebase_features
from sk_classify import run_grid_search

from nltk.stem.porter import *

DATA_PATH = 'data/twitter_davidson/labeled_data_cleaned.csv'

parser = argparse.ArgumentParser(description='Run bag of words baselines.')
parser.add_argument("--clf", default='lr')
parser.add_argument("--scoring", default='roc_auc')
parser.add_argument('-hb', dest='hatebase', action='store_const',
                    const=True, default=False)
parser.add_argument('-stem', dest='stem', action='store_const',
                    const=True, default=False)
parser.add_argument('--word', dest='word', action='store_const',
                    const=True, default=True)
parser.add_argument('--hb_only', dest='hb_only', action='store_const',
                    const=True, default=False)
# parser.add_argument('--ngram', dest='ngram', action='store_const',
#                     const=True, default=False)
# parser.add_argument('--chargram', dest='char', action='store_const',
#                     const=True, default=False)
# parser.add_argument('--tfidf', dest='tfidf', action='store_const',
#                     const=True, default=False)
# parser.add_argument('-a', dest='all', action='store_const',
#                     const=True, default=False)
parser.add_argument('--multi', dest='multi', action='store_const',
                    const=True, default=False)
parser.add_argument('--data', dest='datafile', default=DATA_PATH)
args = parser.parse_args()

#

def basic_tokenizer(sentence):
    tokens = sentence.strip().split() #basic tokenizer
    return [w.rstrip(' ?:!,;.()-_') for w in tokens if w.rstrip(' ?:!,;.()-_')]

def stem_tokenizer(sentence):
    tokens = basic_tokenizer(sentence)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    return stemmed_tokens

#

data = pd.read_csv( args.datafile, header = 0, quoting = 0, 
    dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )
data_raw = data['tweet']

tok = stem_tokenizer if args.stem else basic_tokenizer

if args.hatebase:
    print "Generating hatebase features..."
    hb_features = hatebase_features( data_raw.values.astype('U'), sparse=True, tokenizer=tok )

# Helper functions

def transform_data( vectorizer, data_raw ):
    if args.hb_only:
        return hb_features
    features = vectorizer.fit_transform( data_raw.values.astype('U') )
    if args.hatebase:
        features = hstack((features, hb_features))
    return features

#

if args.word or args.all:

    print "Creating the bag of words..."

    vectorizer = CountVectorizer( analyzer = "word", tokenizer = tok, preprocessor = None, 
        stop_words = None )

    data_x = transform_data(vectorizer, data_raw )

    if args.multi:
        print data_x
        run_grid_search( data_x, data['class'], args.clf, args.scoring )
    else:
        run_grid_search( data_x, data['hate_speech'], args.clf, args.scoring )
#

# if args.ngram or args.all:

#     print "Creating the bag of n=1,2,3 grams..."

#     vectorizer = CountVectorizer( analyzer = "word", ngram_range=(1,3), tokenizer = None, preprocessor = None, 
#         stop_words = None )

#     train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

#     if args.multi:
#         train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
#     else:
#         train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
# #

# if args.char or args.all:

#     print "Creating the bag of character n grams..."

#     vectorizer = CountVectorizer( analyzer = "char_wb", ngram_range=(3,5), tokenizer = None, preprocessor = None, 
#         stop_words = None )

#     train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

#     if args.multi:
#         train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
#     else:
#         train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

# #

# if args.tfidf or args.all:

#     print "Vectorizing: TF-IDF"

#     vectorizer = TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True )

#     train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

#     if args.multi:
#         train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
#     else:
#         train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )


