#!/usr/bin/env python

# Glove vectors + linear regression

import os
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from tqdm import *

GLOVE_DIM = 25
GLOVE_PATH = 'data/glove/glove.twitter.27B.%dd.txt'% GLOVE_DIM
GLOVE_SIZE = 1193514

VOCAB_PATH = 'data/glove/vocab.dat'
DATA_PATH = 'data/twitter_davidson/labeled_data_cleaned.csv'
DATA_SIZE = 26954
EMBED_PATH = 'data/glove/embeddings.%dd.dat' % GLOVE_DIM

parser = argparse.ArgumentParser(description='Run classification with GloVe representations.')
parser.add_argument('--multi', dest='multi', action='store_const',
                    const=True, default=False)
args = parser.parse_args()

data = pd.read_csv( DATA_PATH, header = 0, quoting = 0, 
    dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )

#

def initialize_vocabulary():
    # map vocab to word embeddings
    rev_vocab = []
    for line in open(VOCAB_PATH, 'rb'):
        rev_vocab.append(line.decode('utf-8').strip())
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab

def process_glove(vocab):
    if not os.path.isfile(EMBED_PATH):
        #glove = np.random.randn(len(vocab), GLOVE_DIM)
        glove = np.zeros((len(vocab), GLOVE_DIM))
        with open(GLOVE_PATH, 'r') as fh:
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = list(map(float, array[1:]))
                    glove[idx, :] = vector
        pd.DataFrame(glove).to_csv(EMBED_PATH, header = False, index = False)

def create_vocabulary():
    if not os.path.isfile(VOCAB_PATH):
        print("Creating vocabulary %s from data %s" % (VOCAB_PATH, DATA_PATH))
        vocab = {}
        for line in tqdm(data['tweet'], total=DATA_SIZE):
            tokens = line.strip().split() #basic tokenizer
            for w in tokens:
                w = w.rstrip(' ?:!,;.')
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab))
        with open(VOCAB_PATH, mode="wb") as vocab_file:
            for w in vocab:
                vocab_file.write(w + b"\n")

#

train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

train = data.ix[train_i]
test = data.ix[test_i]

create_vocabulary()
vocab = initialize_vocabulary()
process_glove(vocab)
embeddings = pd.read_csv(EMBED_PATH, header = None, dtype = np.float64)

# Helper function

def transform_data( transform, train_raw, test_raw ):
    vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
        vocabulary = vocab )
    train_counts = vectorizer.transform( train_raw.values.astype('U') )
    print "ok transform train"
    test_counts = vectorizer.transform( test_raw.values.astype('U') )
    print "ok transform test"

    if transform == 'sum':
        print "before dot"
        train_x = np.dot(train_counts, embeddings)
        test_x = np.dot(train_counts, embeddings)
        print train_counts
        print train_x
        return (train_x, test_x)
    if transform == 'ave':
        train_x = np.dot(train_counts, embeddings)/train_counts.sum(axis=1)[:,None]
        test_x = np.dot(train_counts, embeddings)/train_counts.sum(axis=1)[:,None]
        return (train_data_features, test_data_features)
    # if transform == 'bin': # only encodes presence of word, not # occurrences
    #     train_counts = (train_counts > 0).astype(float)
    #     test_counts = (train_counts > 0).astype(float)

#

print "Summing GloVe vectors..."

train_data_features, test_data_features = transform_data('sum', train['tweet'], test['tweet'] )

if args.multi:
    train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
else:
    train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

#

print "Averaging GloVe vectors..."

train_data_features, test_data_features = transform_data('ave', train['tweet'], test['tweet'] )

if args.multi:
    train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
else:
    train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )
