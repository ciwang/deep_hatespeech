from utility import hatebase_features

import os
from os.path import join as pjoin
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

def setup_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data", "twitter_davidson")
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    # parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()

def create_vocabulary( vocab_path, data_raw ):
    if not os.path.isfile(vocab_path):
        print("Creating vocabulary %s from data %s" % (vocab_path, str(data_paths)))
        vocab = {}
        for data in data_raw:
            for line in tqdm(data, total=DATA_SIZE):
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

def initialize_vocabulary(vocab_path):
    # map vocab to word embeddings
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def process_glove( vocab, embed_path, glove_dim ):
    GLOVE_SIZE = 1193514

    if not os.path.isfile(embed_path):
        #glove = np.random.randn(len(vocab), GLOVE_DIM)
        glove = np.zeros((len(vocab), glove_dim))
        with open(GLOVE_PATH, 'r') as fh:
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = list(map(float, array[1:]))
                    glove[idx, :] = vector
        pd.DataFrame(glove).to_csv(embed_path, header = False, index = False)

def counts_to_vec( counts, embeddings ):  
    vecs = []
    for i in tqdm(range(counts.shape[0])):
        result = np.dot(counts[i], embeddings)
        vecs.append(result)
    return np.stack(vecs, axis=0)

def vectorize_data( data_raw, data_vec_path, vectorizer, embeddings ):
    counts = vectorizer.transform( data_raw.values.astype('U') ).toarray()

    # only encodes presence of word, not # occurrences
    data_vec = counts_to_vec( (counts > 0).astype(float), embeddings )

    # concat hatebase features
    print "Generating hatebase features..."
    hatebase_vec = hatebase_features( data_raw.values.astype('U') )

    data_vec = np.concatenate((data_vec, hatebase_vec), axis=1)
    pd.DataFrame(data_vec).to_csv(data_vec_path, header = False, index = False)

#

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.data_dir, "vocab.dat")
    embed_path = pjoin("data", "glove", "embeddings.%dd.dat") % args.glove_dim

    train_raw = pd.read_csv( pjoin(args.data_dir, "train.x"), header = 0, quoting = 0 )['tweet']
    test_raw = pd.read_csv( pjoin(args.data_dir, "test.x"), header = 0, quoting = 0 )['tweet']
    
    create_vocabulary(vocab_path, [train_raw, test_raw])
    vocab = initialize_vocabulary(vocab_path)
    process_glove(vocab, embed_path, args.glove_dim)

    embeddings = pd.read_csv(embed_path, header = None, dtype = np.float64)
    vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
                                    vocabulary = vocab )
    # vectorize and write data
    train_vec_path = pjoin(args.data_dir, "train.%dd.vec" % args.glove_dim)
    test_vec_path = pjoin(args.data_dir, "test.%dd.vec" % args.glove_dim)
    vectorize_data(train_raw, train_vec_path, vectorizer, embeddings)
    vectorize_data(test_raw, test_vec_path, vectorizer, embeddings)
