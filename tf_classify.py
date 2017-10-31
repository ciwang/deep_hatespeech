from tf_regularized_linear_models import LRClassifierL2, SoftmaxClassifier
from utility import train_and_eval_auc

import os
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import json

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
# tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
# tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary. (default 100)")
# tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
# tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("data_dir", "data/twitter_davidson", "SQuAD directory (default ./data/twitter_davidson)")
# tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/twitter_davidson/vocab.dat", "Path to vocab file (default: ./data/twitter_davidson/vocab.dat)")
# tf.app.flags.DEFINE_string("embed_path", "", "Path to the GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the GLoVe embedding (default: ./data/glove/embeddings.{embedding_size}d.dat)")
# tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

# def initialize_model(session, model):
#     logging.info("Created model with fresh parameters.")
#       session.run(tf.global_variables_initializer())
#     logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
#     return model

def load_dataset(x_file, y_file):
    with open(x_file, 'rb') as x_file, open(y_file, 'rb') as y_file:
        data_x = pd.read_csv( x_file, header = None, quoting = 0, dtype = np.float32 )
        data_y = pd.read_csv( y_file, header = 0, quoting = 0, usecols = ['hate_speech'], dtype = np.float32)
        return data_x.values, data_y.values.ravel()

def main(_):
    # embed_path = FLAGS.embed_path or pjoin("data", "glove", "embeddings.{}d.dat".format(FLAGS.embedding_size))
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(pjoin(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    train_x_path = pjoin(FLAGS.data_dir, "train.%dd.vec") % FLAGS.embedding_size
    train_y_path = pjoin(FLAGS.data_dir, "train.y")
    test_x_path = pjoin(FLAGS.data_dir, "test.%dd.vec") % FLAGS.embedding_size
    test_y_path = pjoin(FLAGS.data_dir, "test.y")

    train_x, train_y = load_dataset(train_x_path, train_y_path)
    test_x, test_y = load_dataset(test_x_path, test_y_path) #FLAGS.batch_size

    # lr = SoftmaxClassifier()
    lr = LRClassifierL2(C=100)
    train_and_eval_auc( train_x, train_y, test_x, test_y, model=lr )

if __name__ == "__main__":
  tf.app.run()
