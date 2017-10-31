import numpy as np
import sys
import tensorflow as tf

from utility import HATEBASE_NUM_FIELDS

'''
Attached is some TensorFlow code I wrote last year. The base class will work for all kinds of classifiers, 
and the subclasses are for regular linear classifiers, the second with regularization. I am not sure 
this code still works, since TensorFlow is changing so fast, but maybe this will be a useful start.

The most relevant line is 149. The regularization term is applied to W. If there were multiple Ws, 
one for each group of variables of interest, then one could apply multiple separate terms. You can see 
this already with how I am (non-standardly) regularizing the bias.

The base class could handle a model with multiple Ws. You just need to write a new `build_graph` and 
`train_dict`, modeled on the ones in `SoftmaxClassifier`. Let me know if you try this out!
'''

__author__ = 'Chris Potts'


class TfModelBase(object):
    """
    This base class is common to all of the models defined
    in this tutorial. It handles the nuts-and-bolts of
    optimization and prediction. The `fit` and `predict`
    methods mimic sklearn.
    """
    def __init__(self, max_iter=1000, eta=0.01, tol=1e-4):
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol

    def fit(self, X, y, **kwargs):
        """`X` is an array of predictors. `y` is a label
        vector and will be one-hot encoded. Our only use
        of `kwargs` is to pass `X_test` and `y_test` to
        monitor test performance during training.
        """
        # Preliminaries:
        self.input_dim = X.shape[1]
        # One-hot encoding of target `y`, and creation
        # of a class attribute.
        y = self.prepare_output_data(y)

        # Build the computation graph. This method is
        # instantiated by individual subclasses. It
        # defines the model.
        self.build_graph()

        self.sess = tf.InteractiveSession()

        # Optimizer set-up:
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Training, full dataset for each iteration:
        for i in range(1, self.max_iter+1):
            _, loss = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict=self.train_dict(X, y))
            if loss < self.tol:
                self.progressbar("stopping with loss < self.tol", i)
                return True
            else:
                self.progressbar("loss: {}".format(loss), i)

    def prepare_output_data(self, y):
        self.classes = sorted(set(y))
        self.output_dim = len(self.classes)
        y = self.onehot_encode(y)
        return y

    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it."""
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            labels=self.outputs, logits=self.model))

    def get_optimizer(self):
        return tf.train.GradientDescentOptimizer(
            self.eta).minimize(self.cost)

    def predict_proba(self, X):
        return self.sess.run(
            self.model, feed_dict=self.test_dict(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        return [self.classes[np.argmax(row)] for row in probs]

    def onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_

    def progressbar(self, msg, index, interval=100):
        if index % interval == 0:
            sys.stderr.write('\r')
            sys.stderr.write("Iteration {}: {} ".format(index, msg))
            sys.stderr.flush()

class LRClassifierL2(TfModelBase):
    """
    Extends `TfClassifierBase` with a `build_graph`
    method that defines the linear regression classifier 
    with L2 regularization, as well
    as standard, no-frills `train_dict` and `test_dict`
    methods used by `fit` and `predict_proba`.
    """
    def __init__(self, C=1.0, max_iter=10000, eta=0.01, tol=1e-4):
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
