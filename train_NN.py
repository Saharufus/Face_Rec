from model_building import build_siamese_model
from preprocessing import split_train_test, create_dataset
import tensorflow as tf
import os
import config


gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

anchor = tf.data.Dataset.list_files(os.path.join(config.ANC_PATH, '*.jpg')).take(100)
positive = tf.data.Dataset.list_files(os.path.join(config.POS_PATH, '*.jpg')).take(100)
negative = tf.data.Dataset.list_files(os.path.join(config.NEG_PATH, '*.jpg')).take(100)

data = create_dataset(anchor, positive, negative)
train, test = split_train_test(data, 0.8)
model = build_siamese_model()
