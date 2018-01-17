import os
from os.path import join as pjoin
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from mittens.tf_mittens import Mittens, _log_of_array_ignoring_zeros

def basic_tokenizer(sentence):
    tokens = sentence.strip().split() #basic tokenizer
    return [w.rstrip(' ?:!,;.()-_') for w in tokens if w.rstrip(' ?:!,;.()-_')]

def setup_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data", "twitter_davidson")
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument("--mittens", default=1, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.data_dir, "vocab.dat")
    embed_path = pjoin(args.data_dir, "embeddings.%dd.dat") % args.glove_dim
    mittens_path = pjoin(args.data_dir, "embeddings.mittens%d.%dd.dat") % (args.mittens, args.glove_dim)

    train_raw = pd.read_csv( pjoin(args.data_dir, "train.x"), header = 0, quoting = 0 )['tweet']
    test_raw = pd.read_csv( pjoin(args.data_dir, "test.x"), header = 0, quoting = 0 )['tweet']
    data_raw = pd.concat([train_raw, test_raw])

    # Read vocab
    vocab = []
    with open(vocab_path, mode="rb") as f:
        vocab.extend(f.readlines())
    vocab = [line.strip('\n') for line in vocab]

    print 'Generating co-occurrence matrix ...'
    vectorizer = CountVectorizer( analyzer = "word", tokenizer = basic_tokenizer, preprocessor = None, 
                                    vocabulary = vocab )
    counts = vectorizer.transform( data_raw.values.astype('U') )
    X = counts.T * counts
    X = X.toarray()

    # Read GloVE
    G = pd.read_csv(embed_path, header = None, dtype = np.float64)
    embeddings = G.transpose()
    embeddings.columns = vocab
    embedding_dict = embeddings.to_dict('list')

    print 'Running Mittens with max_iter=%d and mittens=%d' % (args.max_iter, args.mittens)
    mittens = Mittens(n=args.glove_dim, max_iter=args.max_iter, mittens=args.mittens)
    M = mittens.fit(X, vocab=vocab, initial_embedding_dict=embedding_dict)

    print 'Writing embeddings to file %s' % mittens_path
    pd.DataFrame(M).to_csv(mittens_path, header = False, index = False)
