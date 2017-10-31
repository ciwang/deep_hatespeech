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

from utility import train_and_eval_auc
from utility import hatebase_features

DATA_PATH = 'data/twitter_davidson/labeled_data_cleaned.csv'

parser = argparse.ArgumentParser(description='Run bag of words baselines.')
parser.add_argument('--word', dest='word', action='store_const',
                    const=True, default=False)
parser.add_argument('--ngram', dest='ngram', action='store_const',
                    const=True, default=False)
parser.add_argument('--chargram', dest='char', action='store_const',
                    const=True, default=False)
parser.add_argument('--tfidf', dest='tfidf', action='store_const',
                    const=True, default=False)
parser.add_argument('-a', dest='all', action='store_const',
                    const=True, default=False)
parser.add_argument('--multi', dest='multi', action='store_const',
                    const=True, default=False)
parser.add_argument('-hb', dest='hatebase', action='store_const',
                    const=True, default=False)
parser.add_argument('--data', dest='datafile', default=DATA_PATH)
args = parser.parse_args()

#

data = pd.read_csv( args.datafile, header = 0, quoting = 0, 
    dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )

train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

train = data.ix[train_i]
test = data.ix[test_i]
train_raw = train['tweet']
test_raw = test['tweet']

if args.hatebase:
    print "Generating hatebase features..."
    hatebase_train = hatebase_features( train_raw.values.astype('U'), sparse=True )
    hatebase_test = hatebase_features( test_raw.values.astype('U'), sparse=True )

# Helper function

def transform_data( vectorizer, train_raw, test_raw ):
    train_features = vectorizer.fit_transform( train_raw.values.astype('U') )
    test_features = vectorizer.transform( test_raw.values.astype('U') )
    if args.hatebase:
        train_features = hstack((train_features, hatebase_train))
        test_features = hstack((test_features, hatebase_test))
    return (train_features, test_features)

#

if args.word or args.all:

    print "Creating the bag of words..."

    vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
        stop_words = None )

    train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
#

if args.ngram or args.all:

    print "Creating the bag of n=1,2,3 grams..."

    vectorizer = CountVectorizer( analyzer = "word", ngram_range=(1,3), tokenizer = None, preprocessor = None, 
        stop_words = None )

    train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
#

if args.char or args.all:

    print "Creating the bag of character n grams..."

    vectorizer = CountVectorizer( analyzer = "char_wb", ngram_range=(3,5), tokenizer = None, preprocessor = None, 
        stop_words = None )

    train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

#

if args.tfidf or args.all:

    print "Vectorizing: TF-IDF"

    vectorizer = TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True )

    train_data_features, test_data_features = transform_data(vectorizer, train_raw, test_raw )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )


