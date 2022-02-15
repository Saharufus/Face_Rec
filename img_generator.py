import numpy as np
from tensorflow.keras.utils import Sequence
from random import random
from preprocessing import load_positive_pair, load_negative_pair, rescale_resize
import config


class MyImageGenerator(Sequence):
    """Image generator for the face rec NN"""
    def __init__(self, n_pairs=1e+4, batch_size=64, input_size=config.IMG_SHAPE):
        """
        :param n_pairs: The number of pairs to load overall
        :param batch_size: The  batch of pairs to load at once
        :param input_size: The input size of the images
        """
        self.n_pairs = n_pairs
        self.batch_size = batch_size
        self.input_size = input_size

    def __getitem__(self, item):
        """gets the rows of images with a coin flip for positive or negative"""
        anchor = []
        inp = []
        y = []
        for i in range(self.batch_size):
            coin = random()
            # for positive
            if coin > 0.5:
                anc, pos = load_positive_pair()
                anc = rescale_resize(anc)
                pos = rescale_resize(pos)
                anchor.append(anc)
                inp.append(pos)
                y.append(1)
            # for negative
            else:
                anc, neg = load_negative_pair()
                anc = rescale_resize(anc)
                neg = rescale_resize(neg)
                anchor.append(anc)
                inp.append(neg)
                y.append(0)

        return [np.array(anchor), np.array(inp)], np.array(y)

    def on_epoch_end(self):
        # nothing to do when epoch ends because it is random pairs
        pass

    def __len__(self):
        # returns number of batches
        return self.n_pairs // self.batch_size
