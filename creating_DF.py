import numpy as np
import os
import conf as config
import cv2
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from random import sample,seed
import glob
import shutil
import os
seed(10)


def rescale_resize(img):
    return cv2.resize(img, (105, 105)) / 255


def load_from_path(path, img_num):
    print(path)
    dir_list = os.listdir(path)
    if 'negative_cleaned' in path:
        random.shuffle(dir_list)
    else:
        dir_list = sorted(dir_list)
    imgs = []
    for i in range(img_num):
        img = cv2.imread(os.path.join(path, dir_list[i]))
        img = rescale_resize(img)
        imgs.append(img)
    return np.array(imgs)


def load_data(img_num=2221):
    """loads the data to three np.array (anchor, positive, negative)"""
    anchor = load_from_path(config.ANC_PATH, img_num)
    positive = load_from_path(config.POS_PATH, img_num)
    negative = load_from_path(config.NEG_PATH_CLEANED, img_num)

    # creating dataset
    X = np.concatenate([np.array([anchor, positive]), np.array([anchor, negative])], axis=1)
    y = np.concatenate([np.ones(img_num), np.zeros(img_num)], axis=0)

    return X, y


def changing_names(path,name):
    for count, f in enumerate(os.listdir(path)):
        f_name, f_ext = os.path.splitext(f)
        new_name = name + '_' + '0'*(4-len(str(count+1))) + str(count+1) + f_ext
        os.rename(path + '/' + f,path + '/' + new_name)


def add_to_folder(new_path, src_path, dest_path):
    for i in new_path:
        src_dir = src_path + i
        dst_dir = dest_path
        shutil.copy(src_dir, dst_dir)