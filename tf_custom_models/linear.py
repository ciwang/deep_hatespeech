import numpy as np
import sys
import tensorflow as tf

from base import TfModelBase

__author__ = 'Cindy Wang (adapted from Chris Potts)'
HATEBASE_NUM_FIELDS = 8

class LRClassifierL2(TfModelBase):
    """
    Extends `TfClassifierBase` with a `build_graph`
    method that defines the linear regression classifier 
    with L2 regularization, as well
    as standard, no-frills `train_dict` and `test_dict`
    methods used by `fit` and `predict_proba`.
    """
    def __init__(self, C=1.0, max_iter=1000, eta=0.01, tol=1e-4):
        """`C` is the inverse regularization strength."""
        self.C = 1.0 / C
        print "C is ", self.C
        super(LRClassifierL2, self).__init__(
            max_iter=max_iter, eta=eta, tol=tol)

    def fit(self, X, y, **kwargs):
        '''
        Special case where inputs are in groups, so we
        handle the input_dim differently
        '''
        self.input_dim_g = X.shape[1] - HATEBASE_NUM_FIELDS
        self.input_dim_h = HATEBASE_NUM_FIELDS
        super(LRClassifierL2, self).fit(
            X, y, **kwargs)

    def build_graph(self):
        # Input and output placeholders

        self.inputs_g = tf.placeholder(
            tf.float32, shape=[None, self.input_dim_g])
        self.inputs_h = tf.placeholder(
            tf.float32, shape=[None, self.input_dim_h])
        # self.inputs = tf.placeholder(
        #     tf.float32, shape=[None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # Parameters:
        self.W_g = tf.Variable(
            tf.zeros([self.input_dim_g, self.output_dim]))
        self.W_h = tf.Variable(
            tf.zeros([self.input_dim_h, self.output_dim]))
        # self.W = tf.Variable(
        #     tf.zeros([self.input_dim, self.output_dim]))
        self.b = tf.Variable(
            tf.zeros([self.output_dim]))

        # The graph:
        # self.model = tf.matmul(self.inputs, tf.concat([self.W_g, self.W_h], 0)) + self.b
        self.model = tf.matmul(self.inputs_g, self.W_g) + tf.matmul(self.inputs_h, self.W_h) + self.b

    def train_dict(self, X, y):
        X_g = X[:,:-HATEBASE_NUM_FIELDS]
        X_h = X[:,-HATEBASE_NUM_FIELDS:]
        return {self.inputs_g: X_g, self.inputs_h: X_h, self.outputs: y}
        # return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        X_g = X[:,:-HATEBASE_NUM_FIELDS]
        X_h = X[:,-HATEBASE_NUM_FIELDS:]
        return {self.inputs_g: X_g, self.inputs_h: X_h}
        # return {self.inputs: X}

    def get_cost_function(self, **kwargs):
        # This penalizes the bias term too, which is not often done,
        # but this at least shows how to do this for multiple tensors:
        reg = self.C * (tf.nn.l2_loss(self.W_g) + tf.nn.l2_loss(self.W_h)) #try L1 norm
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.outputs, logits=self.model)
        return tf.reduce_mean(reg + cross_entropy)

class SoftmaxClassifier(TfModelBase):
    """
    Extends `TfClassifierBase` with a `build_graph`
    method that defines the softmax classifier, as well
    as standard, no-frills `train_dict` and `test_dict`
    methods used by `fit` and `predict_proba`.
    """
    def __init__(self, max_iter=1000, eta=0.01, tol=1e-4):
        super(SoftmaxClassifier, self).__init__(
            max_iter=max_iter, eta=eta, tol=tol)

    def build_graph(self):
        # Input and output placeholders
        self.inputs = tf.placeholder(
            tf.float32, shape=[None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # Parameters:
        self.W = tf.Variable(
            tf.zeros([self.input_dim, self.output_dim]))
        self.b = tf.Variable(
            tf.zeros([self.output_dim]))

        # The graph:
        self.model = tf.matmul(self.inputs, self.W) + self.b

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}

class SoftmaxClassifierL2(SoftmaxClassifier):
    """
    Exactly the same as `SoftmaxClassifier` but with
    L2-regularization. In TensorFlow, this is applied to
    the cost function as one would expect, and TensorFlow
    takes care of all the details.
    """
    def __init__(self, C=1.0, max_iter=1000, eta=0.01, tol=1e-4):
        """`C` is the inverse regularization strength."""
        self.C = 1.0 / C
        super(SoftmaxClassifierL2, self).__init__(
            max_iter=max_iter, eta=eta, tol=tol)

    def get_cost_function(self, **kwargs):
        # This penalizes the bias term too, which is not often done,
        # but this at least shows how to do this for multiple tensors:
        reg = self.C * (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.outputs, logits=self.model)
        return tf.reduce_mean(reg + cross_entropy)
