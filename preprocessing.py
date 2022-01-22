import numpy as np
import os
import config
import cv2
from sklearn.model_selection import train_test_split


def rescale_resize(img):
    return cv2.resize(img, (105, 105)) / 255


def load_from_path(path, img_num):
    dir_list = os.listdir(path)
    imgs = []
    for i in range(img_num):
        img = cv2.imread(os.path.join(path, dir_list[i]))
        img = rescale_resize(img)
        imgs.append(img)
    return np.array(imgs)


def load_data(img_num=300):
    """loads the data to three np.array (anchor, positive, negative)"""
    anchor = load_from_path(config.ANC_PATH, img_num)
    positive = load_from_path(config.POS_PATH, img_num)
    negative = load_from_path(config.NEG_PATH, img_num)

    # creating dataset
    X = np.concatenate([np.array([anchor, positive]), np.array([anchor, negative])], axis=1)
    y = np.concatenate([np.ones(img_num), np.zeros(img_num)], axis=0)

    return X, y


def train_test_img_split(imgs, labels, train_size=.8):
    ind = range(len(labels))
    train, test = train_test_split(ind, train_size=train_size)
    return imgs[:, train, :, :, :], imgs[:, test, :, :, :], labels[train], labels[test]
