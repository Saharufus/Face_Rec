import preprocessing as pp
import pandas as pd
import tensorflow as tf
import conf as c
import os
import numpy as np
import tensorflow_datasets as tfds
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


def flatten_rows(column):
    """
    This function flattens and deletes the color channel of each sample.
    :param column: column to process
    :return: column processed as a list
    """
    result = []
    for element in column:
        flat = element[0].flatten()
        aux_list = []
        for i,f in enumerate(flat):
            if i%3 == 0:
                aux_list.append(f)
        result.append(aux_list)
    return result


def tf_to_pd_dataframe(df):
    """
    This function coverts a tf dataframe to a pandas dataframe
    :param column: column to process
    :return: column processed as a list
    """

    # first we create an auxiliar list with all the information from the tf dataset
    aux_list = []
    for i, element in enumerate(df):
        aux_list.append(element)

    # We separate the data between anchor, positive and label
    anchor_column = []
    positive_column = []
    label_column = []
    for element in aux_list:
        anchor_column.append(element[c.ANCHOR_CONST])
        positive_column.append(element[c.POSITIVE_CONST])
        label_column.append(element[c.LABEL_CONST])

    # We create the label dataset
    y = np.array(label_column).flatten()

    # We create the X dataset
    # First, we create the rows with the positive values and anchor values
    positive_rows = flatten_rows(positive_column)
    anchor_rows = flatten_rows(anchor_column)

    # Now we merge them
    final_row = []
    for a, p in zip(anchor_rows, positive_rows):
        final_row.append(a + p)

    X_train = pd.DataFrame(final_row)

    return (X_train, y)