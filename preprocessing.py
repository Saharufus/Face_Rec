import tensorflow as tf


# -------------------------------------------------------------------------------------------------
# helper functions
# -------------------------------------------------------------------------------------------------
def scaler():
    """
    scale the img to 105x105 pixels and pix vals [0, 1]
    :param filepath: the path to *.jpg
    :returns: scaled img"""
    # load img
    loaded_img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(loaded_img)
    # resize
    img = tf.image.resize(img, (105, 105))
    # rescale
    img = img / 255
    return img


def label(data1, data2, is_twin=True):
    """
    label the img if same person or not
    :param data1: anchor
    :param data2: positive / negative
    :param is_twin: True / False
    :return: labeled dataset
    """
    # label 1 for same person
    if is_twin:
        return tf.data.Dataset.zip((data1, data2, tf.data.Dataset.from_tensor_slices(tf.ones(len(data1)))))
    # label 0 for different person
    elif not is_twin:
        return tf.data.Dataset.zip((data1, data2, tf.data.Dataset.from_tensor_slices(tf.zeros(len(data1)))))


def prepro_2_img(filepath1, filepath2, is_twin):
    """this function is for mapping the preprocess on the dataset"""
    return {'val image': scaler(filepath1), 'input image': scaler(filepath2)}, is_twin


# -------------------------------------------------------------------------------------------------
# create and preprocess for dataset
# -------------------------------------------------------------------------------------------------
def create_dataset(anchor, positive, negative):
    """
    creates a labeled dataset with positives and negatives
    :param anchor: tf Dataset of anchor img
    :param positive: tf Dataset of positive img
    :param negative: tf Dataset of negative img
    :return: tf Dataset of positive and negative against anchor with label
    """
    # label positive
    positive_labeled = label({'val image': anchor, 'input image': positive}, True)
    # label negative
    negative_labeled = label({'val image': anchor, 'input image': positive}, False)
    # concat, map and cache
    data = positive_labeled.concatenate(negative_labeled)
    data = data.map(prepro_2_img)
    return data


# -------------------------------------------------------------------------------------------------
# train test split
# -------------------------------------------------------------------------------------------------
def split_train_test(data, train_size=0.8, batch_size=16, prefetch_size=8, shuffle=True):
    """
    splits the data into train_data and test_data
    :param data: data to split
    :param train_size: percentage of train out of data
    :param batch_size: the batches size in the data
    :param prefetch_size: data to prepare
    :param shuffle: True or False if shuffle or not
    :return: train_data, test_data
    """
    # shuffling:
    if shuffle:
        data = data.shuffle(buffer_size=len(data)*10)

    # preparing the train data:
    train = data.take(round(len(data)*train_size))
    train = train.batch(batch_size)
    train = train.prefetch(prefetch_size)

    # preparing the test data:
    test = data.skip(round(len(data)*train_size))
    test = test.take(round(len(data)*(1-train_size)))
    test = test.batch(batch_size)
    test = test.prefetch(prefetch_size)

    return train, test
