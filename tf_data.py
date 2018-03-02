from utility import hatebase_features, HATEBASE, HATEBASE_NUM_FIELDS, HATEBASE_FIELDS

import os
from os.path import join as pjoin
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.porter import *
from gensim.models.word2vec import *

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

TWEET_SIZE = 32
stemmer = PorterStemmer()

def setup_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data", "twitter_davidson")
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument('-m', dest='mittens', action='store_const',
                        const='mittens.', default='')
    parser.add_argument('-w', dest='word2vec', action='store_const',
                        const='word2.', default='')
    parser.add_argument('-hb', dest='hb', action='store_const',
                        const='hb.', default='')
    parser.add_argument('-stem', dest='stem', action='store_const',
                        const='stemmed.', default='')
    return parser.parse_args()

def basic_tokenizer(sentence):
    tokens = sentence.strip().split() #basic tokenizer
    return [w.rstrip(' ?:!,;.()-_') for w in tokens if w.rstrip(' ?:!,;.()-_')]

def stem_tokenizer(sentence):
    tokens = basic_tokenizer(sentence)
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    return stemmed_tokens

def create_vocabulary( vocab_path, data_raw, tokenizer=None ):
    if not os.path.isfile(vocab_path):
        print("Creating vocabulary %s" % (vocab_path))
        vocab = {}
        for data in data_raw:
            for line in tqdm(data):
                tokens = tokenizer(line) #basic tokenizer
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

def process_word2vec( vocab, train_raw, embed_path, embed_dim, tokenizer=None ):
    if not os.path.isfile(embed_path):
        print "Writing embeddings to %s" % (embed_path)
        sentences = []
        for line in tqdm(train_raw):
            sentences.append(tokenizer(line)) #stem tokenizer
        model = Word2Vec(sentences, size=embed_dim, hs=1, sg=1, min_count=1, iter=50)
        vecs = np.random.randn(len(vocab), embed_dim)
        for word in tqdm(model.wv.vocab):
            idx = vocab[word]
            vecs[idx, :] = model.wv[word]
        pd.DataFrame(vecs).to_csv(embed_path, header = False, index = False)

def process_with_hatebase( vocab, embeddings, embed_with_hb_path ):
    if not os.path.isfile(embed_with_hb_path):
        hatebase = np.zeros((len(vocab), HATEBASE_NUM_FIELDS))
        with open(HATEBASE,'rb') as hb:
            hatebase_data = pd.read_csv( hb, header = 0, index_col = 0, quoting = 0, 
                                        dtype = HATEBASE_FIELDS, usecols = range(8) )
            hatebase_data = hatebase_data[~hatebase_data.index.duplicated(keep='first')]
            for word in hatebase_data.index:
                if word not in vocab: continue
                idx = vocab[word]
                hatebase[idx, :] = hatebase_data.ix[word]
        embeddings = np.concatenate((embeddings, hatebase), axis=1)
        pd.DataFrame(embeddings).to_csv(embed_with_hb_path, header = False, index = False)

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

def count_vectorize_data( data_raw, data_vec_path, append_hb, vectorizer, embeddings ):
    print "Writing to paths: ", data_vec_path

    counts = vectorizer.transform( data_raw.values.astype('U') ).toarray()

    # only encodes presence of word, not # occurrences
    data_vec = counts_to_vec( (counts > 0).astype(np.float64), embeddings )

    # concat hatebase features
    if append_hb:
        print "Generating hatebase features..."
        hatebase_vec = hatebase_features( data_raw.values.astype('U'), tokenizer=stem_tokenizer )
        data_vec = np.concatenate((data_vec, hatebase_vec), axis=1)
        
    pd.DataFrame(data_vec).to_csv(data_vec_path, header = False, index = False)

def write_coocurr_matrix( data_raw, matrix_path, vectorizer ):
    counts = vectorizer.transform( data_raw.values.astype('U') )
    print "Multiplying ... "
    co_matrix = counts.T * counts
    print "Writing ..."
    pd.DataFrame(co_matrix.toarray()).to_csv(matrix_path, header = False, index = False)

def sentence_to_token_ids(sentence, vocab, tokenizer=None, pad=False):
    words = tokenizer(sentence)
    ids = [vocab.get(w, UNK_ID) for w in words]
    if pad:
        ids = ids[:TWEET_SIZE] + [PAD_ID] * (TWEET_SIZE - min(len(ids), TWEET_SIZE))
    return ids

def data_to_token_ids(data_raw, data_ids_path, vocab, tokenizer=None, pad=False, with_hb=False):
    if not os.path.isfile(data_ids_path):
        print "Generating hatebase features..."
        hatebase_vec = hatebase_features( data_raw.values.astype('U'), tokenizer=tokenizer )
        print("Tokenizing data ...")
        with tf.gfile.GFile(data_ids_path, mode="w") as ids_file:
            counter = 0
            for line in data_raw:
                if counter % 5000 == 0:
                    print("tokenizing line %d" % counter)
                token_ids = sentence_to_token_ids(line, vocab, tokenizer, pad)
                if with_hb:
                    token_ids.extend(hatebase_vec[counter].tolist())
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                else:
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                counter += 1

def data_to_hb(data_raw, hb_vec_path):
    print "Generating hatebase features..."
    hatebase_vec = hatebase_features( data_raw.values.astype('U') )
    pd.DataFrame(hatebase_vec).to_csv(hb_vec_path, header = False, index = False)

#

USE_HB_EMBED = False

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.data_dir, "vocab.%sdat") % args.stem
    embed_path = pjoin(args.data_dir, "embeddings.word2vec.%dd.dat") % args.glove_dim
    if args.mittens:
        embed_path = pjoin(args.data_dir, "embeddings.mittens1.%dd.dat") % args.glove_dim
    glove_path = pjoin("data", "glove", "glove.twitter.27B.%dd.txt") % args.glove_dim
    hb_vocab_path = pjoin("data", "hatebase", "vocab.dat")
    hb_embed_path = pjoin("data", "hatebase", "embeddings.new.%dd.dat") % args.glove_dim

    train_raw = pd.read_csv( pjoin(args.data_dir, "train.x"), header = 0, quoting = 0 )['tweet']
    test_raw = pd.read_csv( pjoin(args.data_dir, "test.x"), header = 0, quoting = 0 )['tweet']
    all_raw = pd.read_csv( pjoin(args.data_dir, "all.x"), header = 0, quoting = 0 )['tweet']
    
    create_vocabulary(vocab_path, [train_raw, test_raw], tokenizer = stem_tokenizer)
    vocab = initialize_vocabulary(vocab_path)

    # write embeddings
    process_glove(vocab, embed_path, glove_path, args.glove_dim)
    process_word2vec(vocab, train_raw, embed_path, args.glove_dim, tokenizer = stem_tokenizer)
    embeddings = pd.read_csv(embed_path, header = None, dtype = np.float64)
    if args.hb:
        embed_hb_path = pjoin(args.data_dir, "embeddings.word2vec.hb.%dd.dat") % args.glove_dim
        embeddings = process_with_hatebase(vocab, embeddings, embed_hb_path)

    # if USE_HB_EMBED:
    #     embeddings = add_hb_embeddings(vocab, embeddings, hb_vocab_path, hb_embed_path)
    #     embed_with_hb_path = pjoin(args.data_dir, "embeddings.withhb.%dd.dat") % args.glove_dim
    #     embeddings.to_csv(embed_with_hb_path, header = False, index = False)

    # NOTE: This block probably obscure now that we're using RNN
    # vectorizer = CountVectorizer( analyzer = "word", tokenizer = stem_tokenizer, preprocessor = None, 
    #                                 vocabulary = vocab )
    # vectorize and write data
    # print "Vectorizing and writing ..."
    # if USE_HB_EMBED:
    #     train_vec_path = pjoin(args.data_dir, "train.withhidden.%dd.vec" % args.glove_dim)
    #     test_vec_path = pjoin(args.data_dir, "test.withhidden.%dd.vec" % args.glove_dim)
    #     all_vec_path = pjoin(args.data_dir, "all.withhidden.%dd.vec" % args.glove_dim)
    # train_vec_path = pjoin(args.data_dir, "train.%dd.%s%s%svec") % (args.glove_dim, args.hb, args.stem, args.word2vec)
    # test_vec_path = pjoin(args.data_dir, "test.%dd.%s%s%svec") % (args.glove_dim, args.hb, args.stem, args.word2vec)
    # all_vec_path = pjoin(args.data_dir, "all.%dd.%s%s%svec") % (args.glove_dim, args.hb, args.stem, args.word2vec)
    # count_vectorize_data(train_raw, train_vec_path, args.hb, vectorizer, embeddings)
    # count_vectorize_data(test_raw, test_vec_path, args.hb, vectorizer, embeddings)
    # count_vectorize_data(all_raw, all_vec_path, args.hb, vectorizer, embeddings)

    # write ids of data
    # train_ids_path = pjoin(args.data_dir, "train.ids.%dd.%s%svec" % (TWEET_SIZE, args.hb, args.stem))
    # test_ids_path = pjoin(args.data_dir, "test.ids.%dd.%s%svec" % (TWEET_SIZE, args.hb, args.stem))
    # data_to_token_ids(train_raw, train_ids_path, vocab, tokenizer=stem_tokenizer, pad=True, with_hb=args.hb)
    # data_to_token_ids(test_raw, test_ids_path, vocab, tokenizer=stem_tokenizer, pad=True, with_hb=args.hb)

    # print "Generating hatebase features..."
    # train_hb_path = pjoin(args.data_dir, "train.hb.vec")
    # test_hb_path = pjoin(args.data_dir, "test.hb.vec")
    # data_to_hb(train_raw, train_hb_path)
    # data_to_hb(test_raw, test_hb_path)
