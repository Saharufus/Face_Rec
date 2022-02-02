import numpy as np
import os
import conf as config
import cv2
import random
from collections import Counter
from random import sample, seed
import glob
import shutil
import os

seed(10)


def rescale_resize(img):
    """
    This function rescales a given image
    :param img: image to resize
    :return: resized image
    """
    return cv2.resize(img, (105, 105)) / 255


def load_from_path(path, nr_rows=56434):
    """
    This function loads from a given path of a folder, all the content to an array
    :param path: the path of the folder
    :param img_num: number of images in the folder
    :return: np array with the content of the folder
    """
    imgs = []
    dir_list = os.listdir(path)

    if 'negative_cleaned' in path:
        for i in range(nr_rows):
            sample = random.sample(dir_list, 1)[0]
            img = cv2.imread(os.path.join(path, sample))
            img = rescale_resize(img)
            imgs.append(img)

    else:
        dir_list = sorted(dir_list)

        img_num = len(dir_list)
        for i in range(img_num):
            if 'anchor' in path:
                img = cv2.imread(os.path.join(path, dir_list[i]))
                img = rescale_resize(img)
                for _ in range(filtered_people[dir_list[i][:-9]]):
                    imgs.append(img)

            elif 'positive' in path:
                name = dir_list[i][:-9]
                if name != "":
                    files = []
                    for name in glob.glob(path + name + '*'):
                        files.append(name)
                    files = sorted(files)
                    for file in files:
                        print(file)
                        if file != "":
                            img = cv2.imread(file)
                            img = rescale_resize(img)
                            imgs.append(img)
                    print("HERE")

    return np.array(imgs)


def load_data(img_num=56434):
    """
    This function loads the data to three np.array (anchor, positive, negative)
    :param img_num: number of images to load (by default 2271)
    :return: a tuple with the np arrays X and y
    """
    anchor = load_from_path(config.ANC_PATH, img_num)
    positive = load_from_path(config.POS_PATH, img_num)
    negative = load_from_path(config.NEG_PATH_CLEANED, img_num)

    # creating dataset
    X = np.concatenate([np.array([anchor, positive]), np.array([anchor, negative])], axis=1)
    y = np.concatenate([np.ones(img_num), np.zeros(img_num)], axis=0)

    return X, y


def changing_names(path,  name):
    """
    This function changes the names of the images we took of ourselves to the same format as in the LFW dataset.
    :param path: the path where the images are stored.
    :param name: the name of the person that appears in the picture.
    """
    for count, f in enumerate(os.listdir(path)):
        f_name, f_ext = os.path.splitext(f)
        new_name = name + '_' + '0' * (4 - len(str(count + 1))) + str(count + 1) + f_ext
        os.rename(path + '/' + f, path + '/' + new_name)


def add_to_folder(new_path, src_path, dest_path):
    """
    This function moves the content of one folder to another one
    :param new_path: a list with the files that we want to change
    :param src_path: source path
    :param dest_path: destination path
    """
    for i in new_path:
        src_dir = src_path + i
        dst_dir = dest_path
        shutil.copy(src_dir, dst_dir)



