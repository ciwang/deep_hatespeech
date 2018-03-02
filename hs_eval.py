import datetime
import os
import time
import logging
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score as F1
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from tf_data import PAD_ID

logging.basicConfig(level=logging.INFO)

def batch(x, y, n=1):
    assert( len(x) == len(y) )
    l = len(x)
    for ndx in range(0, l, n):
        yield (x[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)])

class HateSpeechEval(object):
    def __init__(self, FLAGS, encoder, decoder, embeddings, *args):
        """
        Initializes your System

        :param FLAGS: Tensorflow init flags
        :param args: pass in more arguments as needed
        """
        # ==== constants ==
        self.FLAGS = FLAGS

        # ==== set up placeholder tokens ========
        inputs_size = self.FLAGS.tweet_size
        if self.FLAGS.model_type == 'hb_append':
            inputs_size += self.FLAGS.hatebase_size

        self.inputs_placeholder = tf.placeholder(tf.int32, shape=[None, inputs_size])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])

        # ==== assemble pieces ====
        with tf.variable_scope("hs", initializer=tf.uniform_unit_scaling_initializer(1.0), reuse=tf.AUTO_REUSE):
            self.setup_embeddings(embeddings)
            self.setup_system(encoder, decoder)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.model)

    def setup_system(self, encoder, decoder):
        """
        :return:
        """
        self.H_r = encoder.encode(self.tweets_var)
        if self.FLAGS.model_type == 'hb_append':
            H_hb = tf.cast(tf.slice(self.inputs_placeholder, [0, self.FLAGS.tweet_size], [-1, self.FLAGS.hatebase_size]), tf.float64)
            self.H_r = tf.concat([self.H_r, H_hb], axis = 1)
        self.model = decoder.decode(self.H_r)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with tf.variable_scope("loss"):
            # labels are not one hot encoded
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.model)

    def setup_embeddings(self, embeddings):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            # load data
            if self.FLAGS.model_type == 'hb_append':
                self.tweets_placeholder = tf.slice(self.inputs_placeholder, [0,0], [-1, self.FLAGS.tweet_size])
            else:
                self.tweets_placeholder = self.inputs_placeholder
            self.tweets_var = tf.nn.embedding_lookup(embeddings, self.tweets_placeholder)

    def test(self, session, test_x, test_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}
        input_feed[self.inputs_placeholder] = test_x
        input_feed[self.labels_placeholder] = test_y

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict_proba(self, session, test_x, get_encoding=False):
        
        input_feed = {}
        input_feed[self.inputs_placeholder] = test_x

        output_feed = [tf.nn.softmax(self.model)] # Need to softmax because softmax with cross entropy is used for loss
        if get_encoding:
            output_feed.append(self.H_r)
        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict(self, session, test_x, get_encoding=False):

        outputs = self.predict_proba(session, test_x, get_encoding)
        if get_encoding:
            yp, encod = outputs
            return (np.argmax(yp, axis=1), encod)
        yp = outputs[0]
        return np.argmax(yp, axis=1)

    def validate(self, session, test_x, test_y, log=False):
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
            out = self.test(session, x, y)
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

        if self.FLAGS.scoring == 'auc':
            yp = self.predict_proba(session, sample_x)[0]
            sample_y = np.eye(self.FLAGS.output_size)[sample_y] #one liner for one-hot encoding
            score = AUC(sample_y, yp)
        elif self.FLAGS.scoring == 'f1_macro':
            yp = self.predict(session, sample_x)
            score = F1(sample_y, yp, average='macro')

        if log:
            logging.info("{} - Score: {}, for {} samples".format(dataset_name, score, sample))
        return score
