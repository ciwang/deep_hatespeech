from utility import hatebase_features

import os
from os.path import join as pjoin
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

TWEET_SIZE = 32

def setup_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data", "twitter_davidson")
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    # parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()

def basic_tokenizer(sentence):
    tokens = sentence.strip().split() #basic tokenizer
    return [w.rstrip(' ?:!,;.()-_') for w in tokens if w.rstrip(' ?:!,;.()-_')]

def create_vocabulary( vocab_path, data_raw ):
    if not os.path.isfile(vocab_path):
        print("Creating vocabulary %s" % (vocab_path))
        vocab = {}
        for data in data_raw:
            for line in tqdm(data):
                tokens = basic_tokenizer(line) #basic tokenizer
                for w in tokens:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
        vocab = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab))
        with open(vocab_path, mode="wb") as vocab_file:
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

def process_glove( vocab, embed_path, glove_path, glove_dim ):
    GLOVE_SIZE = 1193514

    if not os.path.isfile(embed_path):
        print "Writing embeddings to %s" % (embed_path)
        #glove = np.zeros((len(vocab), glove_dim))
        glove = np.random.randn(len(vocab), glove_dim)
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = list(map(np.float64, array[1:]))
                    glove[idx, :] = vector
        pd.DataFrame(glove).to_csv(embed_path, header = False, index = False)

def add_hb_embeddings( vocab, embeddings, hb_vocab_path, hb_embed_path ):
    hb_embeddings = pd.read_csv(hb_embed_path, header = None, dtype = np.float64)
    hb_vocab = []
    with tf.gfile.GFile(hb_vocab_path, mode="rb") as f:
        hb_vocab.extend(f.readlines())
    hb_vocab = [line.strip('\n') for line in hb_vocab]
    for i, word in enumerate(hb_vocab):
        if word in vocab:
            print word, "at index", vocab[word]
            #print "old", embeddings.iloc[vocab[word], :]
            #print "new", hb_embeddings.iloc[i, :]
            embeddings.iloc[vocab[word], :] = hb_embeddings.iloc[i, :]
    return embeddings

def counts_to_vec( counts, embeddings ):  
    vecs = []
    for i in tqdm(range(counts.shape[0])):
        result = np.dot(counts[i], embeddings)
        vecs.append(result)
    return np.stack(vecs, axis=0)

def count_vectorize_data( data_raw, data_vec_path, vectorizer, embeddings ):
    counts = vectorizer.transform( data_raw.values.astype('U') ).toarray()

    # only encodes presence of word, not # occurrences
    data_vec = counts_to_vec( (counts > 0).astype(np.float64), embeddings )

    # concat hatebase features
    print "Generating hatebase features..."
    hatebase_vec = hatebase_features( data_raw.values.astype('U') )

    data_vec = np.concatenate((data_vec, hatebase_vec), axis=1)
    pd.DataFrame(data_vec).to_csv(data_vec_path, header = False, index = False)

def write_coocurr_matrix( data_raw, matrix_path, vectorizer ):
    counts = vectorizer.transform( data_raw.values.astype('U') )
    print "Multiplying ... "
    co_matrix = counts.T * counts
    print "Writing ..."
    pd.DataFrame(co_matrix.toarray()).to_csv(matrix_path, header = False, index = False)

def sentence_to_token_ids(sentence, vocab, pad=False):
    words = basic_tokenizer(sentence)
    ids = [vocab.get(w, UNK_ID) for w in words]
    if pad:
        ids = ids[:TWEET_SIZE] + [PAD_ID] * (TWEET_SIZE - min(len(ids), TWEET_SIZE))
    return ids

def data_to_token_ids(data_raw, data_ids_path, vocab, pad=False):
    if not os.path.isfile(data_ids_path):
        print("Tokenizing data ...")
        with tf.gfile.GFile(data_ids_path, mode="w") as ids_file:
            counter = 0
            for line in data_raw:
                counter += 1
                if counter % 5000 == 0:
                    print("tokenizing line %d" % counter)
                token_ids = sentence_to_token_ids(line, vocab, pad)
                ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def data_to_hb(data_raw, hb_vec_path):
    print "Generating hatebase features..."
    hatebase_vec = hatebase_features( data_raw.values.astype('U') )
    pd.DataFrame(hatebase_vec).to_csv(hb_vec_path, header = False, index = False)

#

USE_HB_EMBED = False

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.data_dir, "vocab.dat")
    embed_path = pjoin(args.data_dir, "embeddings.%dd.dat") % args.glove_dim
    glove_path = pjoin("data", "glove", "glove.twitter.27B.%dd.txt") % args.glove_dim
    hb_vocab_path = pjoin("data", "hatebase", "vocab.dat")
    hb_embed_path = pjoin("data", "hatebase", "embeddings.new.%dd.dat") % args.glove_dim

    train_raw = pd.read_csv( pjoin(args.data_dir, "train.x"), header = 0, quoting = 0 )['tweet']
    test_raw = pd.read_csv( pjoin(args.data_dir, "test.x"), header = 0, quoting = 0 )['tweet']
    
    # create_vocabulary(vocab_path, [train_raw, test_raw])
    # vocab = initialize_vocabulary(vocab_path)

    # write embeddings
    # process_glove(vocab, embed_path, glove_path, args.glove_dim)
    # embeddings = pd.read_csv(embed_path, header = None, dtype = np.float64)
    # if USE_HB_EMBED:
    #     embeddings = add_hb_embeddings(vocab, embeddings, hb_vocab_path, hb_embed_path)
    #     embed_with_hb_path = pjoin(args.data_dir, "embeddings.withhb.%dd.dat") % args.glove_dim
    #     embeddings.to_csv(embed_with_hb_path, header = False, index = False)

    # NOTE: This block probably obscure now that we're using RNN
    # vectorizer = CountVectorizer( analyzer = "word", tokenizer = basic_tokenizer, preprocessor = None, 
    #                                 vocabulary = vocab )
    # vectorize and write data
    # print "Vectorizing and writing ..."
    # if USE_HB_EMBED:
    #     train_vec_path = pjoin(args.data_dir, "train.withhidden.%dd.vec" % args.glove_dim)
    #     test_vec_path = pjoin(args.data_dir, "test.withhidden.%dd.vec" % args.glove_dim)
    # else:
    #     train_vec_path = pjoin(args.data_dir, "train.%dd.vec" % args.glove_dim)
    #     test_vec_path = pjoin(args.data_dir, "test.%dd.vec" % args.glove_dim)
    # count_vectorize_data(train_raw, train_vec_path, vectorizer, embeddings)
    # count_vectorize_data(test_raw, test_vec_path, vectorizer, embeddings)

    # write ids of data
    # train_ids_path = pjoin(args.data_dir, "train.ids.%dd.vec" % TWEET_SIZE)
    # test_ids_path = pjoin(args.data_dir, "test.ids.%dd.vec" % TWEET_SIZE)
    # data_to_token_ids(train_raw, train_ids_path, vocab, pad=True)
    # data_to_token_ids(test_raw, test_ids_path, vocab, pad=True)

    print "Generating hatebase features..."
    train_hb_path = pjoin(args.data_dir, "train.hb.vec")
    test_hb_path = pjoin(args.data_dir, "test.hb.vec")
    data_to_hb(train_raw, train_hb_path)
    data_to_hb(test_raw, test_hb_path)
