#!/usr/bin/env python

# BOW + linear regression

import os
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

parser = argparse.ArgumentParser(description='Run bag of words baselines.')
parser.add_argument('--word', dest='word', action='store_const',
					const=True, default=False)
parser.add_argument('--ngram', dest='word', action='store_const',
					const=True, default=False)
parser.add_argument('--chargram', dest='char', action='store_const',
                   	const=True, default=False)
parser.add_argument('--tfidf', dest='tfidf', action='store_const',
                   	const=True, default=False)
parser.add_argument('-a', dest='all', action='store_const',
                   	const=True, default=False)
parser.add_argument('--multi', dest='multi', action='store_const',
                   	const=True, default=False)
args = parser.parse_args()

#

data_file = 'data/twitter_davidson/labeled_data_cleaned.csv'
data = pd.read_csv( data_file, header = 0, quoting = 0, 
	dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )

train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

train = data.ix[train_i]
test = data.ix[test_i]

# Helper function

def train_and_eval_auc( train_x, train_y, test_x, test_y ):
	model = LR()
	model.fit( train_x, train_y )
	p = model.predict_proba( test_x )
	auc = AUC( test_y, p[:,1] )
	print "AUC:", auc

def transform_data( vectorizer, train_raw, test_raw ):
	train_data_features = vectorizer.fit_transform( train_raw.values.astype('U') )
	test_data_features = vectorizer.transform( test_raw.values.astype('U') )
	return (train_data_features, test_data_features)

#

if args.word or args.all:

	print "Creating the bag of words..."

	vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
		stop_words = None )

	train_data_features, test_data_features = transform_data(vectorizer, train['tweet'], test['tweet'] )

	if args.multi:
		train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
	else:
		train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
#

if args.word or args.all:

	print "Creating the bag of n=1,2,3 grams..."

	vectorizer = CountVectorizer( analyzer = "word", ngram_range=(1,3), tokenizer = None, preprocessor = None, 
		stop_words = None )

	train_data_features, test_data_features = transform_data(vectorizer, train['tweet'], test['tweet'] )

	if args.multi:
		train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
	else:
		train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
#

if args.char or args.all:

	print "Creating the bag of character n grams..."

	vectorizer = CountVectorizer( analyzer = "char_wb", ngram_range=(3,5), tokenizer = None, preprocessor = None, 
		stop_words = None )

	train_data_features, test_data_features = transform_data(vectorizer, train['tweet'], test['tweet'] )

	if args.multi:
		train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
	else:
		train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

#

if args.tfidf or args.all:

	print "Vectorizing: TF-IDF"

	vectorizer = TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True )

	train_data_features, test_data_features = transform_data(vectorizer, train['tweet'], test['tweet'] )

	if args.multi:
		train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
	else:
		train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )


