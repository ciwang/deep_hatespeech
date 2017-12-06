from tf_custom_models import OneLayerNNRetrofit
from utility import train_and_eval_auc, HATEBASE_FIELDS
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score as AUC
import scipy.spatial.distance
from operator import itemgetter

import matplotlib.pyplot as plt

import os
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import json
import itertools

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
# tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
# tf.app.flags.DEFINE_integer("state_size", 50, "Size of hidden layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary. (default 100)")
tf.app.flags.DEFINE_string("data_dir", "data/hatebase", "Hatebase directory (default ./data/hatebase)")
tf.app.flags.DEFINE_string("vocab_path", "data/twitter_davidson/vocab.dat", "Path to vocab file (default: ./data/twitter_davidson/vocab.dat)")
tf.app.flags.DEFINE_boolean("force_load_embeddings", False, "Force loading new hatebase embeddings")

def load_embeddings(embed_path, vocab, force=False):
    GLOVE_SIZE = 1193514
    GLOVE_PATH = "data/glove/glove.twitter.27B.%dd.txt" % FLAGS.embedding_size

    if force or not os.path.exists(embed_path):
        hb_vecs = np.zeros((len(vocab), FLAGS.embedding_size))
        with open(GLOVE_PATH, 'r') as fh:
            found = []
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    found.append(idx)
                    vector = list(map(np.float64, array[1:]))
                    hb_vecs[idx, :] = vector
            # words not found are set to random
            # avg = hb_vecs[found, :].mean(axis=0)
            unfound = list(set(vocab.values()) - set(found))
            for i in unfound:
                hb_vecs[i, :] = np.random.randn(FLAGS.embedding_size)
            
        hb_vecs = pd.DataFrame(hb_vecs)
        hb_vecs.to_csv(embed_path, header = False, index = False)
        return hb_vecs, found, unfound

    with open(embed_path, 'rb') as embed_path:
        data_x = pd.read_csv( embed_path, header = None, quoting = 0, dtype = np.float64 )
        return data_x

def get_compare_embeddings(original_embeddings, tuned_embeddings, vocab, dimreduce_type="pca", random_state=0):
    """ Compare embeddings drift. """
    if dimreduce_type == "pca":
        from sklearn.decomposition import PCA
        dimreducer = PCA(n_components=2, random_state=random_state)
    elif dimreduce_type == "tsne":
        from sklearn.manifold import TSNE
        dimreducer = TSNE(n_components=2, random_state=random_state)
    else:
        raise Exception("Wrong dimreduce_type.")

    reduced_original = dimreducer.fit_transform(original_embeddings)
    reduced_tuned = dimreducer.fit_transform(tuned_embeddings)

    def compare_embeddings(word):
        if word not in vocab:
            return None
        word_id = vocab[word]
        original_x, original_y = reduced_original[word_id, :]
        tuned_x, tuned_y = reduced_tuned[word_id, :]
        return original_x, original_y, tuned_x, tuned_y

    return compare_embeddings

def print_embeddings(embeddings_list, vocab):
    '''Takes list of embeddings that have the same indices.
    Each set of embeddings will be plotted in a different color.'''
    for vocab, old, new in zip(vocab, embeddings_list[0], embeddings_list[1]):
        print vocab
        print old
        print new
        print '-----------------'

    tsne = TSNE(n_components=2, random_state=0)
    pca = PCA(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    colors = itertools.cycle(["r", "b", "g"])

    for wv in embeddings_list:
        Y = pca.fit_transform(wv)
     
        plt.scatter(Y[:, 0], Y[:, 1], color=next(colors))
        for label, x, y in zip(vocab, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

def cosine(u, v):        
    """Cosine distance between 1d np.arrays `u` and `v`, which must have 
    the same dimensionality. Returns a float."""
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def neighbors(word, mat, rownames, distfunc=cosine):    
    """Tool for finding the nearest neighbors of `word` in `mat` according 
    to `distfunc`. The comparisons are between row vectors.
    
    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.
        
    mat : np.array
        The vector-space model.
        
    rownames : list of str
        The rownames of mat.
            
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure  
        between 1d vectors.
        
    Raises
    ------
    ValueError
        If word is not in rownames.
    
    Returns
    -------    
    list of tuples
        The list is ordered by closeness to `word`. Each member is a pair 
        (word, distance) where word is a str and distance is a float.
    
    """
    if word not in rownames:
        raise ValueError('%s is not in this VSM' % word)
    w = mat[rownames.index(word)]
    dists = [(rownames[i], distfunc(w, mat[i])) for i in range(len(mat))]
    return sorted(dists, key=itemgetter(1), reverse=False)

def get_multioutput_y(data_y):
    # y transformed from [[0,0,1],[1,0,1]] => [[[1,0],[0,1]],...]
    # one hots in pos i of each vec form y_i
    y = []
    for i in range(len(HATEBASE_FIELDS)):
        y.append([np.eye(2)[vec[i]] for vec in data_y.values])
    return y

def main(_):
    embed_path = pjoin(FLAGS.data_dir, "embeddings.%dd.dat") % FLAGS.embedding_size
    new_embed_path = pjoin(FLAGS.data_dir, "embeddings.new.%dd.dat") % FLAGS.embedding_size
    hb_path = pjoin(FLAGS.data_dir, "lexicon.csv")

    hatebase_data = pd.read_csv( hb_path, header = 0, index_col = 0, quoting = 0, 
                                    dtype = HATEBASE_FIELDS, usecols = range(len(HATEBASE_FIELDS)+1) )
    vocab = dict([(x, y) for (y, x) in enumerate(hatebase_data.index)])
    hatebase_embeddings = load_embeddings(embed_path, vocab, FLAGS.force_load_embeddings)

    # print neighbors("bitch", hatebase_embeddings.values, list(hatebase_data.index.values), cosine)[:5]
    # print neighbors("hoe", hatebase_embeddings.values, list(hatebase_data.index.values), cosine)[:5]
    # print neighbors("redneck", hatebase_embeddings.values, list(hatebase_data.index.values), cosine)[:5]
    
    train_i, test_i = train_test_split( np.arange( len( hatebase_embeddings )), train_size = 0.8, random_state = 44 )
    train_x = hatebase_embeddings.ix[train_i].values
    test_x = hatebase_embeddings.ix[test_i].values
    train_y = get_multioutput_y(hatebase_data.ix[train_i])
    test_y = get_multioutput_y(hatebase_data.ix[test_i])

    nn = OneLayerNNRetrofit(h=200, max_iter=4000, retrofit_iter=2000)
    new_embeddings = nn.fit( train_x, train_y )

    probs = nn.predict_proba( test_x )

    total_auc = 0
    for i in range(len(probs)):
        # hack to get the positive class
        y = [s[1] for s in test_y[i]]
        y_pred = [s[1] for s in probs[i]]
        auc = AUC( y, y_pred )
        print HATEBASE_FIELDS.keys()[i], "AUC:", auc
        total_auc += auc
    print "Average AUC:", total_auc/len(probs)

    new_embeddings = pd.DataFrame(new_embeddings)
    new_embeddings.to_csv(new_embed_path, header = False, index = False)
    # hidden_states = nn.return_hidden_states( hatebase_embeddings )

    # print neighbors("bitch", hidden_states, list(hatebase_data.index.values), cosine)[:5]
    # print neighbors("hoe", hidden_states, list(hatebase_data.index.values), cosine)[:5]
    # print neighbors("redneck", hidden_states, list(hatebase_data.index.values), cosine)[:5]
    #print_embeddings( [hatebase_embeddings.values, hidden_states], vocab, 50 )
    #print_embeddings( [hatebase_embeddings.ix[unfound_i].values, hidden_states[unfound_i, :]], unfound_vocab )
    # print_embeddings( [hatebase_embeddings.ix[found_i].values, hidden_states[found_i, :]], found_vocab )

if __name__ == "__main__":
    tf.app.run()
