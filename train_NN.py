from model_building import build_siamese_model, one_vgg16_stream
from img_generator import MyImageGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from model_building import DistLayer


def train_model(epochs=200, batch_size=16, patience=None, model_exists=False, lr=1e-4, model_name='model.h5'):
    """trains the NN for face recognition"""
    gpus = tf.config.list_physical_devices('GPU')
    # checking for GPU on device
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except IndexError:
        pass

    # setting train and validation generators
    train = MyImageGenerator(n_pairs=10000, batch_size=batch_size, input_size=(224, 224))
    val = MyImageGenerator(n_pairs=2500, batch_size=batch_size, input_size=(224, 224))

    # loads model if exists (building if not)
    if model_exists:
        model = tf.keras.models.load_model(model_name, custom_objects={'DistLayer': DistLayer})
    else:
        model = build_siamese_model(one_vgg16_stream(trainable=True))
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics='accuracy')

    # if early stopping is set
    if patience:
        es = [EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True)]
    else:
        es = None

    try:
        model.fit(
            train,
            validation_data=val,
            epochs=epochs,
            callbacks=es)
    except KeyboardInterrupt():
        pass
    # saves the model
    finally:
        model.save(model_name)


if __name__ == '__main__':
    train_model(batch_size=8, model_exists=True, lr=1e-6, epochs=50, model_name='FR_v1.h5', patience=20)
