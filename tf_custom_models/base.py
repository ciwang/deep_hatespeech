import numpy as np
import sys
import tensorflow as tf

from tensorflow.python import debug as tf_debug

__author__ = 'Cindy Wang (adapted from Chris Potts)'

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
        if len(y.shape) == 1 or y.shape[1] == 1:
            self.classes = sorted(set(y))
            self.output_dim = len(self.classes)
            y = np.eye(self.output_dim)[y] # one liner shortcut
            #y = self.onehot_encode(y)
        else:
            self.output_dim = y.shape[1]
            self.classes = range(self.output_dim)
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
            tf.nn.softmax(self.model), feed_dict=self.test_dict(X))

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
