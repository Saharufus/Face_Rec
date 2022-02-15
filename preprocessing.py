import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from random import randint
import config


# ---------------------------------------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------------------------------------

def rescale_resize(img):
    """rescale the image"""
    return cv.resize(img, config.IMG_SHAPE) / 255


def get_img(path):
    """converts path of image to RGB numpy array"""
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def load_positive_pair():
    """loads a positive pair of images"""
    data_list = os.listdir('data')
    # get a random image
    anc_file = data_list[randint(0, len(data_list) - 1)]
    anc_name = anc_file[:-9]
    anc_num = anc_file[-8:-4]

    # get a random pair from same person directory
    anc_dir_list = os.listdir(os.path.join('lfw', anc_name))
    pos_file = anc_dir_list[randint(0, len(anc_dir_list) - 1)]
    pos_num = pos_file[-8:-4]
    # shuffle until different image
    while pos_num == anc_num:
        pos_file = anc_dir_list[randint(0, len(anc_dir_list)-1)]
        pos_num = pos_file[-8:-4]

    anc_path = os.path.join('data', anc_file)
    pos_path = os.path.join('data', pos_file)
    return get_img(anc_path), get_img(pos_path)


def load_negative_pair():
    """loads a negative pair of images"""
    data_list = os.listdir('data')
    # get a random image
    anc_file = data_list[randint(0, len(data_list) - 1)]
    anc_name = anc_file[:-9]

    # get a another random image
    neg_file = data_list[randint(0, len(data_list) - 1)]
    neg_name = neg_file[:-9]
    # shuffle until different person
    while neg_name == anc_name:
        neg_file = data_list[randint(0, len(data_list) - 1)]
        neg_name = neg_file[:-9]

    anc_path = os.path.join('data', anc_file)
    neg_path = os.path.join('data', neg_file)
    return get_img(anc_path), get_img(neg_path)


# ---------------------------------------------------------------------------------------------------------------
# main functions
# ---------------------------------------------------------------------------------------------------------------

def load_all_from_path(path):
    """loads every file from path and return it as numpy array"""
    dir_list = os.listdir(path)
    imgs = []
    for file in dir_list:
        img = get_img(os.path.join(path, file))
        img = rescale_resize(img)
        imgs.append(img)
    return np.array(imgs)


def create_dataset(num_of_samples):
    """creates dataset with 2*num_os_samples rows of images"""
    imgs = []

    # get positive
    for i in range(num_of_samples):
        anc, pos = load_positive_pair()
        anc = rescale_resize(anc)
        pos = rescale_resize(pos)
        imgs.append([anc, pos])

    # get negative
    for i in range(num_of_samples):
        anc, neg = load_negative_pair()
        anc = rescale_resize(anc)
        neg = rescale_resize(neg)
        imgs.append([anc, neg])

    # get labels
    labels = np.concatenate([np.ones(num_of_samples), np.zeros(num_of_samples)])
    return np.array(imgs), labels


def train_test_img_split(imgs, labels, train_size=.8):
    """splits data to train and test"""
    ind = range(len(labels))
    train, test = train_test_split(ind, train_size=train_size)
    return imgs[train, :, :, :, :], imgs[test, :, :, :, :], labels[train], labels[test]
