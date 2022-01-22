from model_building import build_siamese_model
from preprocessing import load_data, train_test_img_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import config
import numpy as np


gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_img_split(X, y)

model = build_siamese_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics='accuracy')
model.fit([X_train[0, :, :, :, :], X_train[1, :, :, :, :]], y_train, epochs=10, batch_size=32, validation_split=.1)
