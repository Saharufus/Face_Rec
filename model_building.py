from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf


def one_image_stream():
    """Builds a one image stream model"""
    # input
    inp = Input(shape=(105, 105, 3), name='input layer')

    # first block
    conv1 = Conv2D(64, (10, 10), activation='relu', name='conv1')(inp)
    max1 = MaxPooling2D(64, (2, 2), padding='same', name='max_pool1')(conv1)

    # second block
    conv2 = Conv2D(128, (7, 7), activation='relu', name='conv2')(max1)
    max2 = MaxPooling2D(64, (2, 2), padding='same', name='max_pool2')(conv2)

    # third block
    conv3 = Conv2D(128, (4, 4), activation='relu', name='conv3')(max2)
    max3 = MaxPooling2D(64, (2, 2), padding='same', name='max_pool3')(conv3)

    # forth block
    conv4 = Conv2D(256, (4, 4), activation='relu', name='conv4')(max3)
    flat = Flatten(name='flatten1')(conv4)
    dense = Dense(4096, activation='sigmoid', name='dense1')(flat)

    return Model(inputs=[inp], outputs=[dense], name='one_image_stream')


class DistLayer(Layer):
    """Layer that compares dist between two images"""
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, val_embedding):
        return tf.math.square(input_embedding - val_embedding)


def build_siamese_model():
    """Build a classifier NN"""
    # input image
    input_img = Input(shape=(105, 105, 3), name='input image')

    # anchor image
    val_img = Input(shape=(105, 105, 3), name='val image')

    # combine
    embedding = one_image_stream()
    dist = DistLayer(name='distance')(embedding(input_img), embedding(val_img))

    # classifying
    classifier = Dense(1, activation='sigmoid', name='classifier')(dist)

    return Model(inputs=[input_img, val_img], outputs=[classifier], name='SiameseNetwork')
