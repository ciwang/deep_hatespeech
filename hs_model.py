import datetime
import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from tf_data import PAD_ID
from hs_eval import HateSpeechEval

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    return tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)

class Encoder(object):
    def __init__(self, state_size, embedding_size, num_layers, input_dropout, output_dropout, state_dropout):
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.input_keep_prob = 1 - input_dropout
        self.output_keep_prob = 1 - output_dropout
        self.state_keep_prob = 1 - state_dropout

    def encode(self, inputs, reuse=False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial hidden state input into this function.

        :param inputs: Tweets, with shape [batch_size, tweet_size, embed_size]
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: Encodings of each tweet. A tensor of shape [batch_size, state_size]
        """
        # symbolic function takes in Tensorflow object, returns tensorflow object

        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            stacked_rnn = []
            for i in range(self.num_layers):
                cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, 
                            input_keep_prob=self.input_keep_prob,
                            output_keep_prob=self.output_keep_prob,
                            state_keep_prob=self.state_keep_prob,
                            variational_recurrent=True,
                            input_size=self.embedding_size if i == 0 else self.state_size,
                            dtype=tf.float64)
                stacked_rnn.append(cell)
            self.cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
            # _, (_, m_state) = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=masks, dtype=tf.float64)
            _, final_state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=length(inputs), dtype=tf.float64)

        _, final_m_state = final_state[-1] # get the final state from the last hidden layer
        return final_m_state

class Decoder(object):
    def __init__(self, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

    def decode(self, inputs):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param inputs: Hidden reps of each tweet with shape [batch_size, state_size]
        :return: Probability distribution over classes
        """
        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", shape=(self.state_size, self.output_size),
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            b = tf.get_variable("b", shape=(self.output_size),
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

        return tf.matmul(inputs, W) + b


class HateSpeechSystem(object):
    def __init__(self, FLAGS, train_x, train_y, *args):
        """
        Initializes your System

        :param FLAGS: Tensorflow init flags
        :param args: pass in more arguments as needed
        """
        # ==== constants ====
        self.FLAGS = FLAGS
        self.tweets = tf.constant(train_x)
        self.labels = tf.constant(train_y)
        tweet, label = tf.train.slice_input_producer(
            [self.tweets, self.labels], num_epochs=self.FLAGS.epochs)

        # ==== set up placeholder tokens ========
        self.tweets_placeholder, self.labels_placeholder = tf.train.batch(
            [tweet, label], batch_size=self.FLAGS.batch_size)

        # ==== assemble pieces ====
        with tf.variable_scope("hs", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_training_op()

        self.setup_eval()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        self.encoder = Encoder(self.FLAGS.state_size, 
                            self.FLAGS.embedding_size, 
                            self.FLAGS.num_layers, 
                            self.FLAGS.input_dropout,
                            self.FLAGS.output_dropout,
                            self.FLAGS.state_dropout)
        H_r = self.encoder.encode(self.tweets_var)

        if self.FLAGS.model_type == 'hb_append':
            self.decoder = Decoder(self.FLAGS.state_size + self.FLAGS.hatebase_size, self.FLAGS.output_size)
            self.model = self.decoder.decode(H_r)
        else:
            self.decoder = Decoder(self.FLAGS.state_size, self.FLAGS.output_size)
            self.model = self.decoder.decode(H_r)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # labels are not one hot encoded
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.model)

    def setup_training_op(self):
        optimizer = get_optimizer(self.FLAGS.optimizer)(self.FLAGS.learning_rate)
        gradients, variables = map(list, zip(*optimizer.compute_gradients(self.loss)))
        self.grad_norm = tf.global_norm(gradients)
        # gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :self.tweets_var: tf.Variable with same shape as inputs [batch_size, tweet_size, embed_size]
        """
        with vs.variable_scope("embeddings"):
            # load data
            glove_matrix = pd.read_csv(self.FLAGS.embed_path, header = None, dtype = np.float64)
            embeddings = tf.Variable(glove_matrix, trainable=False)
            self.tweets_var = tf.nn.embedding_lookup(embeddings, self.tweets_placeholder)

    def setup_eval(self):
        self.eval = HateSpeechEval(self.FLAGS, self.encoder, self.decoder)

    def train(self, session, train_x, train_y, test_x, test_y):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        logging.info("Evaluating initial")
        val_loss = self.eval.validate(session, test_x, test_y, log=True)
        self.eval.evaluate_answer(session, train_x, train_y, "Train", log=True)
        self.eval.evaluate_answer(session, test_x, test_y, "Validation", log=True)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        saver = tf.train.Saver()

        # Start the training loop.
        try:
            num_processed = 0
            curr_epoch = 0
            best_train = (0, 0) # epoch, auc
            best_test = (0, 0)
            while not coord.should_stop():
                tic = time.time()
                _, loss, grad_norm = session.run([self.train_op, self.loss, self.grad_norm])
                num_processed += 1

                toc = time.time()
                if (num_processed % self.FLAGS.print_every == 0):
                    logging.info("Epoch = %d | Num batches processed = %d | Train epoch ETA = %f | Grad norm = %f | Training loss = %f" % 
                        (curr_epoch, num_processed, (self.FLAGS.save_every - num_processed) * (toc - tic), grad_norm, np.mean(loss)))
                
                if (num_processed % self.FLAGS.save_every == 0):  
                    num_processed = 0
                    curr_epoch += 1

                    results_path = os.path.join(self.FLAGS.train_dir, "results/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now()))
                    model_path = results_path + "model.weights/"
                    if not os.path.exists(model_path):
            		    os.makedirs(model_path)
                    save_path = saver.save(session, model_path)
                    logging.info("Model saved in file: %s" % save_path)

                    logging.info("Evaluating epoch %d", curr_epoch)
                    val_loss = self.eval.validate(session, test_x, test_y, log=True)
                    train_auc = self.eval.evaluate_answer(session, train_x, train_y, "Train", log=True)
                    if train_auc >= best_train[1]:
                        best_train = (curr_epoch, train_auc)
                    test_auc = self.eval.evaluate_answer(session, test_x, test_y, "Validation", log=True)
                    if test_auc >= best_test[1]:
                        best_test = (curr_epoch, test_auc)

                    logging.info("Best train: Epoch %d AUC: %f" % best_train)
                    logging.info("Best test: Epoch %d AUC: %f" % best_test)
                    if best_test[1] - test_auc >= 0.1:
                        logging.info("Test error has diverged. Halting training.")
                        break
        except tf.errors.OutOfRangeError:
            print('Saving')
            saver.save(sess, FLAGS.train_dir, global_step=num_processed)
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()