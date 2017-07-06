# -*- coding: utf-8 -*-
"""
build DNNRegressor model for owvlab data and calculate the accuracy of the model

@author: Wang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf
import itertools
import numpy as np

#feature list
COMP_ATTR = ['C1','C1_value','Rb','Ground0','Rc','Rc_value','C2','C2_value','RL','RL_value',
'DC0','DC0_on','DC0_value','Signal','Sig_wav','Sig_freq','Sig_amp','Q1','DCam0','DCam0_on','DCam1','DCam1_on',
'ACvolt0','ACvolt0_on','OCS','OSC_on','DCvolt0','DCvolt0_on']
CONN_ATTR = ['RL-C2','RL-GND','Sig-GND','DC0-GND','Sig-C1','Rc-DC0','Q1-GND','Rc-C2','Rc-DCam0','Rb-DC0','Q1-DCam0',
'Rb-DCam1','C1-DCam1','Q1-DCam1','ACvolt0-GND','OCS_ch1-GND','OCS_ch2-GND','OCS-Sig','OCS-RL','ACvolt0-RL','DCvolt0-GND','DCvolt0-Q1']
# data sets
OWVLAB_TRAINGING = "training_data.csv"
OWVLAB_TEST = "student_data.csv"

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = COMP_ATTR + CONN_ATTR + ['Result']
FEATURES = COMP_ATTR + CONN_ATTR
LABEL = "Result"


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels

def dist(x):
    if abs(x[0]-x[1]) < 5.:
        return 1
    else:
        return 0

def main(unused_argv):
    # Load datasets
    training_set = pd.read_csv(OWVLAB_TRAINGING, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv(OWVLAB_TEST, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]

    # Build 2 layer fully connected DNN with 50, 50 units respectively.
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[50,50],
                                              model_dir="./owvlab_model")

    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=10000)
    
    y = regressor.predict(input_fn=lambda: input_fn(test_set))

    # .predict() returns an iterator; convert to a list and print predictions

    test_set_ = pd.read_csv(OWVLAB_TEST, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
    test_set_label = list(test_set_[LABEL])
    predictions = list(itertools.islice(y, len(test_set_label)))
    print ("Predictions: {}".format(str(predictions)))
    pred_score = list(map(dist,zip(test_set_label,predictions)))
    print (pred_score)
    print ("Accuracy: %g" % (sum(pred_score)/len(pred_score)))

if __name__ == "__main__":
    tf.app.run()
