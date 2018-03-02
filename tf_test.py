from utility import hatebase_features
from utility.twitter_preprocess import clean

import os
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import json

from hs_model import HateSpeechSystem
from tf_data import *

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_type", "classic", "classic / hb_embed / hb_append")
tf.app.flags.DEFINE_boolean("bidirectional", False, "Bidirectionality of LSTM")
tf.app.flags.DEFINE_string("scoring", "f1_macro", "auc / f1_macro")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("input_dropout", 0.0, "Fraction of input units randomly dropped.")
tf.app.flags.DEFINE_float("output_dropout", 0.0, "Fraction of output units randomly dropped.")
tf.app.flags.DEFINE_float("state_dropout", 0.0, "Fraction of units randomly dropped on recurrent connections.")
tf.app.flags.DEFINE_float("embedding_dropout", 0.0, "Fraction of embeddings randomly dropped on lookup.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 1000, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of encoder layers.")
tf.app.flags.DEFINE_integer("tweet_size", 32, "The length of a tweet (example).")
tf.app.flags.DEFINE_integer("output_size", 3, "The output size, aka number of classes.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary. (default 100)")
tf.app.flags.DEFINE_integer("hatebase_size", 7, "Number of hatebase features. (default 8)")
tf.app.flags.DEFINE_string("data_dir", "data/twitter_davidson", "SQuAD directory (default ./data/twitter_davidson)")
tf.app.flags.DEFINE_string("train_dir", "train", "CHECKPOINT directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 1200, "How many iterations to do per save checkpoint and evaluate.")
tf.app.flags.DEFINE_string("vocab_path", "data/twitter_davidson/vocab.dat", "Path to vocab file (default: ./data/twitter_davidson/vocab.dat)")
tf.app.flags.DEFINE_string("vocab_stemmed", False, "Whether to use stemmed vocabulary (False for GloVe, True for others)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the GLoVe embedding (default: ./data/twitter_davidson/embeddings.{embedding_size}d.dat)")
tf.app.flags.DEFINE_string("embed_trainable", False, "Whether to train embeddings (False for GloVe, True for others)")
tf.app.flags.DEFINE_string("test_path", "", "Path to file with sentences to test (Default empty is REPL)")

def initialize_model(session, model, train_dir, seed=42):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    ckpt = tf.train.get_checkpoint_state(train_dir)
    print ckpt
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    logging.info(str(vars(FLAGS)))
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        # index to word
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # word to index
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def load_dataset(x_file, y_file):
    with open(x_file, 'rb') as x_file, open(y_file, 'rb') as y_file:
        data_x = np.loadtxt( x_file, delimiter = ' ', dtype = int)

        if FLAGS.output_size == 2:
            data_y = pd.read_csv( y_file, header = 0, quoting = 0, usecols = ['hate_speech'], dtype = int)
        elif FLAGS.output_size == 3:
            data_y = pd.read_csv( y_file, header = 0, quoting = 0, usecols = ['class'], dtype = int)
        data_y = data_y.values.ravel()

        return data_x, data_y

def raw_to_ids(lines, vocab, tok):
    lines = [clean(l) for l in lines]
    ids = [sentence_to_token_ids(l, vocab, tokenizer=tok, pad=True) for l in lines]
    # if FLAGS.model_type == 'hb_append':
    #     hb = hatebase_features( lines, tokenizer=tok )
    #     ids.extend(hb)
    return ids

def main(_):
    if FLAGS.embed_path:
        FLAGS.embed_path = FLAGS.embed_path.format(FLAGS.embedding_size)
    else:
        FLAGS.embed_path = pjoin("data", "twitter_davidson", "embeddings.{}d.dat".format(FLAGS.embedding_size))
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    # print(vars(FLAGS))
    with open(pjoin(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    hb = 'hb.' if FLAGS.model_type == 'hb_append' else ''
    stemmed = 'stemmed.' if FLAGS.vocab_stemmed else ''

    train_x_path = pjoin(FLAGS.data_dir, "train.ids.{}d.{}{}vec".format(FLAGS.tweet_size, hb, stemmed))
    test_x_path = pjoin(FLAGS.data_dir, "test.ids.{}d.{}{}vec".format(FLAGS.tweet_size, hb, stemmed))
    print train_x_path, test_x_path
    train_y_path = pjoin(FLAGS.data_dir, "train.y")
    test_y_path = pjoin(FLAGS.data_dir, "test.y")

    train_x, train_y = load_dataset(train_x_path, train_y_path)
    test_x, test_y = load_dataset(test_x_path, test_y_path) #FLAGS.batch_size

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path) # one is list and one is dict
    FLAGS.vocab_len = len(vocab)
    tok = stem_tokenizer if FLAGS.vocab_stemmed else basic_tokenizer

    hs = HateSpeechSystem( FLAGS, train_x, train_y )

    with tf.Session() as sess:
        initialize_model(sess, hs, FLAGS.train_dir)
        if not FLAGS.test_path:
            while True:
                token_ids = raw_to_ids([raw_input("Type sentence: ")], vocab, tok)
                print "Tokens:", token_ids
                proba = hs.eval.predict_proba(sess, token_ids)[0][0]
                print "Probability dist:", proba
                print "Prediction:", np.argmax(proba)
        raw_lines = pd.read_csv( FLAGS.test_path, header = 0, quoting = 0 )['tweet']
        token_ids = raw_to_ids(raw_lines, vocab, tok)
        pred, encod = hs.eval.predict(sess, token_ids, get_encoding=True)
        pd.DataFrame(pred).to_csv( FLAGS.test_path.split('.')[0] + '.pred.csv')
        pd.DataFrame(encod).to_csv( FLAGS.test_path.split('.')[0] + '.encod.csv')

if __name__ == "__main__":
  tf.app.run()
