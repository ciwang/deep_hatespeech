import datetime
import os
import time
import logging
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from tf_data import PAD_ID

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def batch(x, y, n=1):
    assert( len(x) == len(y) )
    l = len(x)
    for ndx in range(0, l, n):
        yield (x[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)])

class HateSpeechEval(object):
    def __init__(self, FLAGS, encoder, decoder, *args):
        """
        Initializes your System

        :param FLAGS: Tensorflow init flags
        :param args: pass in more arguments as needed
        """
        # ==== constants ====
        self.FLAGS = FLAGS

        # ==== set up placeholder tokens ========
        self.tweets_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.tweet_size])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])

        # ==== assemble pieces ====
        with tf.variable_scope("hs", initializer=tf.uniform_unit_scaling_initializer(1.0), reuse=tf.AUTO_REUSE):
            self.setup_embeddings()
            self.setup_system(encoder, decoder)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.model)

    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        H_r = encoder.encode(self.tweets_var)
        self.model = decoder.decode(H_r)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # labels are not one hot encoded
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.model)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            # load data
            glove_matrix = pd.read_csv(self.FLAGS.embed_path, header = None, dtype = np.float64)
            embeddings = tf.Variable(glove_matrix, trainable=False)
            self.tweets_var = tf.nn.embedding_lookup(embeddings, self.tweets_placeholder)

    def test(self, session, test_x, test_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        input_feed[self.tweets_placeholder] = test_x

        input_feed[self.labels_placeholder] = test_y

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict_proba(self, session, test_x):
        
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.tweets_placeholder] = test_x

        output_feed = [tf.nn.softmax(self.model)] # Need to softmax because softmax with cross entropy is used for loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict(self, session, test_x):

        yp = self.predict_proba(session, test_x)

        return np.argmax(yp, axis=1)

    def validate(self, sess, test_x, test_y, log=False):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function (at end of each epoch)

        :return:
        """
        valid_cost = 0
        num_seen = 0

        out = {}
        for x, y in batch(test_x, test_y, self.FLAGS.batch_size):
            out = self.test(sess, x, y)
            valid_cost += sum(out[0])
            num_seen += len(out[0])

        average_valid_cost = float(valid_cost) / float(num_seen)

        if log:
            logging.info("Validate cost: {}".format(average_valid_cost))

        return valid_cost

    def evaluate_answer(self, session, data_x, data_y, dataset_name, sample=4000, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        count = 0
        random_indices = random.sample(xrange(len(data_x)), sample)
        sample_x = data_x[random_indices]
        sample_y = data_y[random_indices]

        yp = self.predict_proba(session, sample_x)[0]
        sample_y = np.eye(self.FLAGS.output_size)[sample_y] #one liner for one-hot encoding
        auc = AUC(sample_y, yp)

        if log:
            logging.info("{} - AUC: {}, for {} samples".format(dataset_name, auc, sample))
        return auc
