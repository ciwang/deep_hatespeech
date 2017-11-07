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
        # self.W_1 = tf.Variable(
        #     tf.random_uniform([self.input_dim, self.h_dim]))
        # self.W_2 = tf.Variable(
        #     tf.random_uniform([self.h_dim, self.output_dim]))
        # self.b_1 = tf.Variable(
        #     tf.zeros([self.h_dim]))
        # self.b_2 = tf.Variable(
        #     tf.zeros([self.output_dim]))

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
        yhat = tf.nn.relu(tf.matmul(self.h, W_2) + b_2)  # No activation
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

