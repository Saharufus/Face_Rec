import numpy as np
import os
import config
import cv2
from sklearn.model_selection import train_test_split


def load_data(img_num=300):
    """loads the data to three np.array (anchor, positive, negative)"""
    # setting lists to fill
    anchor = []
    positive = []
    negative = []

    # getting img filepath
    anc_file_list = os.listdir(config.ANC_PATH)
    pos_file_list = os.listdir(config.POS_PATH)
    neg_file_list = os.listdir(config.NEG_PATH)

    for i in range(img_num):
        # getting the image
        anc_img = cv2.imread(os.path.join(config.ANC_PATH, anc_file_list[i]))
        pos_img = cv2.imread(os.path.join(config.POS_PATH, pos_file_list[i]))
        neg_img = cv2.imread(os.path.join(config.NEG_PATH, neg_file_list[i]))

        # resize and rescale
        anc_img = cv2.resize(anc_img, (105, 105)) / 255
        pos_img = cv2.resize(pos_img, (105, 105)) / 255
        neg_img = cv2.resize(neg_img, (105, 105)) / 255

        # filling
        anchor.append(anc_img)
        positive.append(pos_img)
        negative.append(neg_img)

    # creating dataset
    X = np.concatenate([np.array([anchor, positive]), np.array([anchor, negative])], axis=1)
    y = np.concatenate([np.ones(img_num), np.zeros(img_num)], axis=0)

    return X, y


def train_test_img_split(imgs, labels, train_size=.8):
    ind = range(len(labels))
    train, test = train_test_split(ind, train_size=train_size)
    return imgs[:, train, :, :, :], imgs[:, test, :, :, :], labels[train], labels[test]
