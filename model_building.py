from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import config


# ---------------------------------------------------------------------------------------------------------------
# Siamese model
# ---------------------------------------------------------------------------------------------------------------

class DistLayer(Layer):
    """Layer that compares dist between two images"""
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, val_embedding, input_embedding):
        return tf.math.abs(input_embedding - val_embedding)


def build_siamese_model(one_stream):
    # two inputs:
    anc = Input(shape=(config.IMG_SHAPE, config.IMG_SHAPE, 3))
    inp = Input(shape=(config.IMG_SHAPE, config.IMG_SHAPE, 3))

    # one stream:
    emb = one_stream

    # distance layer:
    dist = DistLayer(name='distance')(emb(anc), emb(inp))

    # classifying:
    classifier = Dense(1, activation='sigmoid', name='classifier')(dist)

    return Model(inputs=[anc, inp], outputs=[classifier])


# ---------------------------------------------------------------------------------------------------------------
# One streams
# ---------------------------------------------------------------------------------------------------------------

def one_youtube_stream():
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


def one_vgg16_stream():
    inp = Input(shape=(224, 224, 3))
    vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    vgg_model.trainable = False
    vgg_model = vgg_model(inp)
    flat = Flatten()(vgg_model)
    return Model(inputs=[inp], outputs=[flat])


def my_one_stream():
    inp = Input(shape=(100, 100, 3), name='Input')

    # first block
    conv11 = Conv2D(32, (7, 7), activation='relu', name='conv11')(inp)
    conv12 = Conv2D(32, (5, 5), activation='relu', name='conv12')(conv11)
    conv13 = Conv2D(64, (3, 3), activation='relu', name='conv13')(conv12)
    max1 = MaxPooling2D((2, 2))(conv13)

    # second block
    conv21 = Conv2D(128, (7, 7), activation='relu', name='conv21')(max1)
    conv22 = Conv2D(128, (5, 5), activation='relu', name='conv22')(conv21)
    conv23 = Conv2D(256, (3, 3), activation='relu', name='conv23')(conv22)
    max2 = MaxPooling2D((2, 2))(conv23)

    # third block
    conv3 = Conv2D(512, (7, 7), activation='relu', name='conv3')(max2)
    flatten = Flatten(name='flatten')(conv3)
    dense = Dense(256, activation='relu', name='dense')(flatten)

    return Model(inputs=[inp], outputs=[dense], name='MyOneStream')
