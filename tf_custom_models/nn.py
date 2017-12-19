import numpy as np
import sys
import tensorflow as tf

from base import TfModelBase

__author__ = 'Cindy Wang (adapted from vinhkhuc on Github)'

class OneLayerNN(TfModelBase):
    # h is the hidden layer size
    def __init__(self, max_iter=1000, eta=0.01, tol=1e-4, h=50):
        super(OneLayerNN, self).__init__(
            max_iter=max_iter, eta=eta, tol=tol)
        self.h_dim = h

    def build_graph(self):
        # Input and output placeholders
        self.inputs = tf.placeholder(
            tf.float64, shape=[None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float64, shape=[None, self.output_dim])

        # Parameters:
        self.W_1 = tf.get_variable("W_1", shape=(self.input_dim, self.h_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        self.W_2 = tf.get_variable("W_2", shape=(self.h_dim, self.output_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        self.b_1 = tf.get_variable("b_1", shape=(self.h_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        self.b_2 = tf.get_variable("b_2", shape=(self.output_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

        # The graph:
        self.model = self._forwardprop(self.inputs, self.W_1, self.W_2, self.b_1, self.b_2)

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}

    def _forwardprop(self, inputs, W_1, W_2, b_1, b_2):
        """
        Forward-propagation.
        """
        self.h = tf.nn.sigmoid(tf.matmul(inputs, W_1) + b_1)  # The \sigmoid function
        yhat = tf.nn.softmax(tf.matmul(self.h, W_2) + b_2)  # Need activation since not using softmax cross entropy
        return yhat

    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it."""
        return tf.losses.mean_squared_error( \
            labels=self.outputs, predictions=self.model)

    def predict_proba(self, X):
        return self.sess.run(
            self.model, feed_dict=self.test_dict(X))

    def return_hidden_states(self, X):
        """Must be called after the model has already been trained
        Outputs hidden layer, which has dimensions [None, h_dim]"""
        return self.sess.run(
            self.h, feed_dict=self.test_dict(X))

class OneLayerNNRetrofit(TfModelBase):
    # for multi binary output ONLY
    # supports retrofitting input vectors

    def __init__(self, max_iter=1000, retrofit_iter=1000, eta=0.001, tol=1e-4, h=50):
        super(OneLayerNNRetrofit, self).__init__(
            max_iter=max_iter, eta=eta, tol=tol)
        self.h_dim = h
        self.retrofit_iter = retrofit_iter

    def fit(self, X, y, **kwargs):
        """Returns retrofitted X, not very clean but will have to do for now
        """
        # Preliminaries:
        self.input_dim = X.shape[1]
        # One-hot encoding of target `y`, and creation
        # of a class attribute.
        self.output_dim = len(y)

        # Build the computation graph. This method is
        # instantiated by individual subclasses. It
        # defines the model.
        self.build_graph()

        self.sess = tf.InteractiveSession()

        # Optimizer set-up:
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()
        self.gradients = tf.gradients(self.cost, self.inputs)

        self.sess.run(tf.global_variables_initializer())

        # Training, full dataset for each iteration:
        for i in range(1, self.max_iter+1):
            _, loss, gradients = self.sess.run(
                [self.optimizer, self.cost, self.gradients],
                feed_dict=self.train_dict(X, y))
            if i > self.max_iter - self.retrofit_iter:
                X -= gradients[0]*self.eta #make more efficient
                print X
            if loss < self.tol:
                self.progressbar("stopping with loss < self.tol", i)
                return True
            else:
                self.progressbar("loss: {}".format(loss), i)

        return X

    def build_graph(self):
        # Input and output placeholders
        self.inputs = tf.placeholder(
            tf.float64, shape=[None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float64, shape=[self.output_dim, None, 2])

        # Parameters:
        self.W_1 = tf.get_variable("W_1", shape=(self.input_dim, self.h_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        self.b_1 = tf.get_variable("b_1", shape=(self.h_dim),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

        self.W_2 = []
        self.b_2 = []
        for i in range(self.output_dim):
            self.W_2.append(tf.get_variable('W_2_%d' % i, shape=(self.h_dim, 2),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64))
            self.b_2.append(tf.get_variable('b_2_%d' % i, shape=(2),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64))

        # The graph:
        self.model = [self._forwardprop(self.inputs, self.W_1, self.W_2[i], self.b_1, self.b_2[i]) \
                        for i in range(self.output_dim)] # model is shape (output_dim, None, 2)

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}

    def _forwardprop(self, inputs, W_1, W_2, b_1, b_2):
        """
        Forward-propagation.
        """
        self.h = tf.nn.sigmoid(tf.matmul(inputs, W_1) + b_1)  # The \sigmoid function
        yhat = tf.matmul(self.h, W_2) + b_2  # Need activation since not using softmax cross entropy
        return yhat

    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it."""
        return sum([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
            labels=self.outputs[i], logits=self.model[i])) for i in range(self.output_dim)])

    def predict(self, X):
        pass

    def predict_proba(self, X):
        return self.sess.run(
            tf.nn.softmax(self.model), feed_dict=self.test_dict(X))

