import numpy as np
import os
import pandas as pd
import conf
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


def df_with_path(path, nr_rows, filtered_people):
    """
    This function loads from a given path of a folder, all the content to an array
    :param path: the path of the folder
    :param img_num: number of images in the folder
    :param filtered_people: a dictionary with all the names of the people in the dataset and the number of images they've got
    :return: np array with the content of the folder
    """
    paths = []
    dir_list = os.listdir(path)

    if 'negative_cleaned' in path:
        for i in range(nr_rows):
            sample = random.sample(dir_list, 1)[0]
            paths.append(sample)

    else:
        dir_list = sorted(dir_list)

        img_num = len(dir_list)
        for i in range(img_num):
            if 'anchor' in path:
                for _ in range(filtered_people[dir_list[i][:-9]]):
                    paths.append(dir_list[i])

            elif 'positive' in path:
                name = dir_list[i][:-9]
                if name != "":
                    files = []
                    for name in glob.glob(path + name + '*'):
                        files.append(name)
                    files = sorted(files)
                    for file in files:
                        if file != "":
                            paths.append(file.split("/")[-1])

    return np.array(paths)


def load_data():
    """
    This function appends the path of the images to three lists (anchor, positive, negative)
    :return: a tuple with the lists
    """
    anchor = df_with_path(conf.ANC_PATH, conf.NR_ROWS, conf.FILTERED_DICT)
    print(anchor.shape)
    positive = df_with_path(conf.POS_PATH, conf.NR_ROWS, conf.FILTERED_DICT)
    print(positive.shape)
    negative = df_with_path(conf.NEG_PATH_CLEANED, conf.NR_ROWS, conf.FILTERED_DICT)
    print(negative.shape)
    return (anchor, positive, negative)


def create_df(anchor, positive, negative):
    """
    This function creates a df with the paths given in the lists
    :param anchor: list with all the paths of the anchor images
    :param positive: list with all the paths of the positive images
    :param negative: list with all the paths of the negative images
    :return: the final df
    """
    df_1 = pd.DataFrame()
    df_1['img_1'] = anchor
    df_1['img_2'] = positive
    df_1['label'] = np.ones(conf.NR_ROWS)

    df_2 = pd.DataFrame()
    df_2['img_1'] = anchor
    df_2['img_2'] = negative
    df_2['label'] = np.zeros(conf.NR_ROWS)

    df = pd.concat([df_1, df_2])
    return df.reset_index().drop(columns=['index'])


def delete_bad_labels(df):
    """
    This function verifies and fixes that the images are correctly labeled
    :param df: the dataframe with the paths
    :return: the fixed df
    """
    aux = 0
    for index, row in df.iterrows():
        if row.label == 0:
            if row.img_1[:-9] == row.img_2[:-9]:
                df.drop(index, inplace=True)
                aux += 1
    return df


def save_df(df):
    """
    This function saves the df into a csv file
    :param df: the df to save
    """
    df.to_csv('dataframe.csv')


def changing_names(path, name):
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
