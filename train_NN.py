from model_building import build_siamese_model
from preprocessing import split_train_test, create_dataset
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import config
import numpy as np


gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

anchor = tf.data.Dataset.list_files(os.path.join(config.ANC_PATH, '*.jpg')).take(300)
positive = tf.data.Dataset.list_files(os.path.join(config.POS_PATH, '*.jpg')).take(300)
negative = tf.data.Dataset.list_files(os.path.join(config.NEG_PATH, '*.jpg')).take(300)

data = create_dataset(anchor, positive, negative)
train, test = split_train_test(data, 0.8, batch_size=16)
model = build_siamese_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics='accuracy')
model.fit(train, epochs=10)

import matplotlib.pyplot as plt
sample = test.unbatch().batch(64).take(1)
# print((model.predict(sample) > 0.5).astype(int))
print((np.array(list(sample.as_numpy_iterator())[0][1]) == (model.predict(sample) > 0.5).astype(int)).mean())

