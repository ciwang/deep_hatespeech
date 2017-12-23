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

class Encoder(object):
    def __init__(self, state_size, embedding_size, num_layers, dropout):
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.keep_prob = 1-dropout

    def encode(self, inputs, masks, reuse=False):
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

        stacked_rnn = []
        for _ in range(self.num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob, seed=42)
            stacked_rnn.append(cell)
        self.cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)

        with tf.variable_scope('rnn'):
            # _, (_, m_state) = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=masks, dtype=tf.float64)
            _, final_state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=masks, dtype=tf.float64)

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
        with tf.variable_scope('softmax'):
            W = tf.get_variable("W", shape=(self.state_size, self.output_size),
                            initializer=tf.contrib.layers.xavier_initializer(seed=42), dtype=tf.float64)
            b = tf.get_variable("b", shape=(self.output_size),
                            initializer=tf.contrib.layers.xavier_initializer(seed=42), dtype=tf.float64)

        return tf.matmul(inputs, W) + b


class HateSpeechSystem(object):
    def __init__(self, FLAGS, *args):
        """
        Initializes your System

        :param FLAGS: Tensorflow init flags
        :param args: pass in more arguments as needed
        """
        # ==== constants ====
        self.FLAGS = FLAGS

        # ==== set up placeholder tokens ========
        self.tweets_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.tweet_size])
        self.masks_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_training_op()

        # ==== set up training/updating procedure ====
        pass

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoder = Encoder(self.FLAGS.state_size, self.FLAGS.embedding_size, self.FLAGS.num_layers, self.FLAGS.dropout)
        H_r = encoder.encode(self.tweets_var, self.masks_placeholder)

        decoder = Decoder(self.FLAGS.state_size, self.FLAGS.output_size)
        self.model = decoder.decode(H_r)

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
        :return:
        """
        with vs.variable_scope("embeddings"):
            # load data
            glove_matrix = pd.read_csv(self.FLAGS.embed_path, header = None, dtype = np.float64)
            embeddings = tf.Variable(glove_matrix, trainable=False)
            self.tweets_var = tf.nn.embedding_lookup(embeddings, self.tweets_placeholder)

    def optimize(self, session, train_x, train_y, masks):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.tweets_placeholder] = train_x
        input_feed[self.masks_placeholder] = masks

        input_feed[self.labels_placeholder] = train_y

        output_feed = [self.train_op, self.loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y, masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        input_feed[self.tweets_placeholder] = valid_x
        input_feed[self.masks_placeholder] = masks

        input_feed[self.labels_placeholder] = valid_y

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict_proba(self, session, test_x, masks):
        
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.tweets_placeholder] = test_x
        input_feed[self.masks_placeholder] = masks

        output_feed = [tf.nn.softmax(self.model)] # Need to softmax because softmax with cross entropy is used for loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict(self, session, test_x, masks):

        yp = self.predict_proba(session, test_x, masks)

        return np.argmax(yp, axis=1)

    def validate(self, sess, valid_dataset, log=False):
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
        for x, y in valid_dataset:
            x_lengths = [min(len(tweet), self.FLAGS.tweet_size) for tweet in x]
            x = [tweet[:self.FLAGS.tweet_size] + [PAD_ID] * (self.FLAGS.tweet_size - min(len(tweet), self.FLAGS.tweet_size)) for tweet in x]
            out = self.test(sess, x, y, x_lengths)
            valid_cost += sum(out[0])
            num_seen += len(out[0])

        average_valid_cost = float(valid_cost) / float(num_seen)

        if log:
            logging.info("Validate cost: {}".format(average_valid_cost))

        return valid_cost

    def evaluate_answer(self, session, dataset, dataset_name, rev_vocab, sample=4000, log=False):
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
        sample_x = []
        sample_x_lengths = []
        sample_y = []

        sample_batches = sample/self.FLAGS.batch_size
        random_indices = random.sample(xrange(len(dataset)), sample_batches)
        for index in random_indices:
            if count >= sample_batches: break
            x, y = dataset[index]

            x_lengths = [min(len(tweet), self.FLAGS.tweet_size) for tweet in x]
            x = [tweet[:self.FLAGS.tweet_size] + [PAD_ID] * (self.FLAGS.tweet_size - min(len(tweet), self.FLAGS.tweet_size)) for tweet in x]
            sample_x += x
            sample_x_lengths += x_lengths
            sample_y += y
            count += 1

        yp = self.predict_proba(session, sample_x, sample_x_lengths)[0]
        sample_y = np.eye(self.FLAGS.output_size)[sample_y] #one liner for one-hot encoding
        auc = AUC(sample_y, yp)

        if log:
            logging.info("{} - AUC: {}, for {} samples".format(dataset_name, auc, sample))
        return auc

    def train(self, session, dataset, val_dataset, train_dir, rev_vocab):
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

        random.seed(42)
        print random.random()
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        logging.info("Evaluating initial")
        val_loss = self.validate(session, val_dataset, log=True)
        self.evaluate_answer(session, dataset, "Train", rev_vocab, log=True)
        self.evaluate_answer(session, val_dataset, "Validation", rev_vocab, log=True)

        # split into train and test loops?
        num_processed = 0
        best_train = (0, 0) # epoch, auc
        best_test = (0, 0)
        for e in range(self.FLAGS.epochs):
            random.shuffle(dataset)
            for x, y in dataset:
                tic = time.time()
                x_lengths = [min(len(tweet), self.FLAGS.tweet_size) for tweet in x]
                x = [tweet[:self.FLAGS.tweet_size] + [PAD_ID] * (self.FLAGS.tweet_size - min(len(tweet), self.FLAGS.tweet_size)) for tweet in x]
                _, loss, grad_norm = self.optimize(session, x, y, x_lengths)
                num_processed += 1
                toc = time.time()
                if (num_processed % self.FLAGS.print_every == 0):
                    logging.info("Epoch = %d | Num batches processed = %d | Train epoch ETA = %f | Grad norm = %f | Training loss = %f" % (e, num_processed, (len(dataset) - num_processed) * (toc - tic), grad_norm, np.mean(loss)))
                
            # save the model
            num_processed = 0
            saver = tf.train.Saver()
            results_path = os.path.join(self.FLAGS.train_dir, "results/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now()))
            model_path = results_path + "model.weights/"
            if not os.path.exists(model_path):
    		    os.makedirs(model_path)
            save_path = saver.save(session, model_path)
            logging.info("Model saved in file: %s" % save_path)
            logging.info("Evaluating epoch %d", e)
            val_loss = self.validate(session, val_dataset, log=True)
            train_auc = self.evaluate_answer(session, dataset, "Train", rev_vocab, log=True)
            if train_auc >= best_train[1]:
                best_train = (e, train_auc)
            test_auc = self.evaluate_answer(session, val_dataset, "Validation", rev_vocab, log=True)
            if test_auc >= best_test[1]:
                best_test = (e, test_auc)

            logging.info("Best train: Epoch %d AUC: %f" % best_train)
            logging.info("Best test: Epoch %d AUC: %f" % best_test)
            if best_test[0] - e > 5:
                logging.info("Best test epoch has not changed for over 5 epochs. Halting training.")
                break

