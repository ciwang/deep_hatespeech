from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import csv
import os
import pandas as pd
import numpy as np

# HATEBASE = os.path.join(os.path.dirname(__file__), '../data/hatebase/lexicon.stemmed.csv')
HATEBASE = os.path.join(os.path.dirname(__file__), '../data/hatebase/lexicon.csv')
HATEBASE_NUM_FIELDS = 7
HATEBASE_FIELDS = {
    'about_class': np.int32, 
    'about_religion': np.int32, 
    'about_gender': np.int32,
    'about_ethnicity': np.int32, 
    'about_nationality': np.int32, 
    'about_sexual_orientation': np.int32, 
    'about_disability': np.int32, 
    'offensiveness': float
    # 'number_of_sightings': np.int32
}

def train_and_eval_auc( train_x, train_y, test_x, test_y, model=LR() ):
    model.fit( train_x, train_y )
    p = model.predict_proba( test_x )
    print p
    # hack
    p = p[:,1] if p.shape[1] > 1 else p[:,0]

    auc = AUC( test_y, p )
    print "AUC:", auc

# Returns matrix of shape (n_examples, n_features)
def hatebase_features( raw_x, tokenizer=None, sparse=False ):
    with open(HATEBASE,'rb') as hb:
        hatebase_data = pd.read_csv( hb, header = 0, index_col = 0, quoting = 0, 
                                    dtype = HATEBASE_FIELDS, usecols = range(8) ) # dtype = HATEBASE_FIELDS, usecols = range(8) )

    def get_feature_vec( str ):
        tokens = tokenizer(str)
        # matches = [hatebase_data[w] for w in hatebase_data if w in tokens] # gets unique tokens
        matches = [w for w in tokens if w in hatebase_data.index]
        feature_vec = hatebase_data.ix[matches].sum(axis=0).values.astype(int)
        if sparse:
            return csr_matrix(feature_vec) 
        return feature_vec

    feature_vecs = [get_feature_vec(str) for str in raw_x]
    if sparse:
        return vstack(feature_vecs)
    return np.stack(feature_vecs, axis=0)

