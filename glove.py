#!/usr/bin/env python

# Glove vectors + linear regression

import os
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from utility import train_and_eval_auc
from utility import hatebase_features

from tqdm import *

parser = argparse.ArgumentParser(description='Run classification with GloVe representations.')
parser.add_argument('--multi', dest='multi', action='store_const',
                    const=True, default=False)
parser.add_argument('-hb', dest='hatebase', action='store_const',
                    const=True, default=False)
parser.add_argument('-d', dest='glove_dim', default=100, type=int)
parser.add_argument('--sum', dest='sum', action='store_const',
                    const=True, default=False)
parser.add_argument('--ave', dest='ave', action='store_const',
                    const=True, default=False)
parser.add_argument('--uniq', dest='uniq', action='store_const',
                    const=True, default=False)
parser.add_argument('--norm', dest='norm', action='store_const',
                    const=True, default=False)
parser.add_argument('-a', dest='all', action='store_const',
                    const=True, default=False)
args = parser.parse_args()

GLOVE_PATH = 'data/glove/glove.twitter.27B.%dd.txt' % args.glove_dim
GLOVE_SIZE = 1193514

VOCAB_PATH = 'data/twitter_davidson/vocab.dat'
DATA_PATH = 'data/twitter_davidson/labeled_data_cleaned.csv'
DATA_SIZE = 26954
EMBED_PATH = 'data/glove/embeddings.%dd.dat' % args.glove_dim

data = pd.read_csv( DATA_PATH, header = 0, quoting = 0, 
    dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )

# Pretrained vector processing

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
        glove = np.zeros((len(vocab), args.glove_dim))
        with open(GLOVE_PATH, 'r') as fh:
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = list(map(np.float64, array[1:]))
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
train_raw = train['tweet']
test_raw = test['tweet']
del(data)

create_vocabulary()
vocab = initialize_vocabulary()
process_glove(vocab)
embeddings = pd.read_csv(EMBED_PATH, header = None, dtype = np.float64)

if args.hatebase:
    print "Generating hatebase features..."
    hatebase_train = hatebase_features( train_raw.values.astype('U') )
    hatebase_test = hatebase_features( test_raw.values.astype('U') )

# Helper functions

def counts_to_vec( counts ):
    vecs = []
    for i in tqdm(range(counts.shape[0])):
        result = np.dot(counts[i], embeddings)
        vecs.append(result)
    return np.stack(vecs, axis=0)

def vectorize_data( train_raw, test_raw ):
    vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
                                    vocabulary = vocab )
    train_counts = vectorizer.transform( train_raw.values.astype('U') ).toarray()
    test_counts = vectorizer.transform( test_raw.values.astype('U') ).toarray()
    return (train_counts, test_counts)

train_x, test_x = None, None
def transform_data( transform, train_counts, test_counts ):
    global train_x, test_x

    if transform == 'sum':
        train_x = counts_to_vec(train_counts)
        test_x = counts_to_vec(test_counts)

    if transform == 'ave':
        if train_x is None and test_x is None:
            train_x = counts_to_vec(train_counts)
            test_x = counts_to_vec(test_counts)
        train_x = train_x/train_counts.sum(axis=1)[:,None]
        test_x = test_x/test_counts.sum(axis=1)[:,None]

    if transform == 'uniq': # only encodes presence of word, not # occurrences
        train_x = counts_to_vec( (train_counts > 0).astype(np.float64) )
        test_x = counts_to_vec( (test_counts > 0).astype(np.float64) )

    if args.norm:
        train_x = normalize(train_x)
        test_x = normalize(test_x)

    if args.hatebase:
        train_x = np.concatenate((train_x, hatebase_train), axis=1)
        test_x = np.concatenate((test_x, hatebase_test), axis=1)

    return (train_x, test_x)
#

print "Vectorizing raw data..."
train_counts, test_counts = vectorize_data(train_raw, test_raw)

if args.sum or args.all:

    print "Summing GloVe vectors..."

    train_data_features, test_data_features = transform_data('sum', train_counts, test_counts )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

#

if args.ave or args.all:

    print "Averaging GloVe vectors..."

    train_data_features, test_data_features = transform_data('ave', train_counts, test_counts )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

#

if args.uniq or args.all:

    print "Summing GloVe vectors (presence only)..."

    train_data_features, test_data_features = transform_data('uniq', train_counts, test_counts )

    if args.multi:
        train_and_eval_auc( train_data_features, train['class'], test_data_features, test['class'].values )
    else:
        train_and_eval_auc( train_data_features, train['hate_speech'], test_data_features, test['hate_speech'].values )

