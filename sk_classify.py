#!/usr/bin/env python

# For scikit classification on distributed representations

import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
import argparse
from time import time

from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC as SVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as AUC

def setup_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data", "twitter_davidson")
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--clf", default='lr')
    parser.add_argument("--scoring", default='roc_auc')
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument('-m', dest='mittens', action='store_const',
                        const='mittens.', default='')
    parser.add_argument('-w', dest='word2vec', action='store_const',
                        const='word2.', default='')
    parser.add_argument('-hb', dest='hb', action='store_const',
                        const='hb.', default='')
    parser.add_argument('-stem', dest='stem', action='store_const',
                        const='stemmed.', default='')
    parser.add_argument('--multi', dest='multi', action='store_const',
                    const=True, default=False)
    return parser.parse_args()

def load_dataset(x_file, y_file, multi):
    with open(x_file, 'rb') as x_file, open(y_file, 'rb') as y_file:
        data_x = pd.read_csv( x_file, header = None, quoting = 0, dtype = np.float64 )
        usecols = ['class'] if multi else ['hate_speech']
        data_y = pd.read_csv( y_file, header = 0, quoting = 0, usecols = usecols, dtype = np.int32)
        return data_x.values, data_y.values.ravel()

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def run_grid_search(data_x, data_y, clf_name, scoring):
    if clf_name == 'lr':
        print "Classifier: Logistic regression"
        clfs = [LR()]
        param_grid = {"C": [0.1, 0.25, 0.5, 1]}

    elif clf_name == 'rf':
        print "Classifier: Random forest"
        clfs = [RF(n_estimators=20, max_depth=20, criterion='entropy', bootstrap=False, 
                    max_features=20, min_samples_split=4, min_samples_leaf=4)]
        # use a full grid over all parameters
        param_grid = {"max_features": [20, 25],
                      "min_samples_split": [2, 3, 4],
                      "min_samples_leaf": [2, 3, 4]}
    elif clf_name == 'svm':
        print "Classifier: SVM"
        clfs = [SVM(penalty='l2', loss='squared_hinge', dual=False)]
        param_grid = {"C": [0.001, 0.1, 1, 5]}
    else:
        print "Unsupported classifier:", clf_name

    # run grid search
    for clf in clfs:
        print clf
        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring=scoring)
        start = time()
        grid_search.fit(data_x, data_y)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_['params'])))
        report(grid_search.cv_results_)

if __name__ == '__main__':
    args = setup_args()

    # if args.mittens:
    #     train_x_path = pjoin(args.data_dir, "train.mittens.%dd.vec") % args.embed_dim
    #     test_x_path = pjoin(args.data_dir, "test.mittens.%dd.vec") % args.embed_dim
    # else:
    #     train_x_path = pjoin(args.data_dir, "train.%dd.vec") % args.embed_dim
    #     test_x_path = pjoin(args.data_dir, "test.%dd.vec") % args.embed_dim
    # train_y_path = pjoin(args.data_dir, "train.y")
    # test_y_path = pjoin(args.data_dir, "test.y")

    all_x_path = pjoin(args.data_dir, "all.%dd.%s%s%svec") % (args.embed_dim, args.hb, args.stem, args.word2vec)
    all_y_path = pjoin(args.data_dir, "all.y")
    print "Running 3-fold CV on", all_x_path

    # train_x, train_y = load_dataset(train_x_path, train_y_path)
    # test_x, test_y = load_dataset(test_x_path, test_y_path)
    all_x, all_y = load_dataset(all_x_path, all_y_path, args.multi)

    run_grid_search(all_x, all_y, args.clf, args.scoring)

    # print model.fit( train_x, train_y ).score( test_x, test_y )
    # p = model.predict_proba( test_x )
    # print p
    # # hack
    # p = p[:,1] if p.shape[1] > 1 else p[:,0]

    # auc = AUC( test_y, p )
    # print "AUC:", auc
