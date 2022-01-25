from model_building import build_siamese_model
from preprocessing import load_data, train_test_img_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from model_building import DistLayer


def train_model(epochs, batch_size, model_exists=False):
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.set_visible_devices(gpu, 'GPU')
    tf.config.experimental.set_memory_growth(gpu, True)

    X, y = load_data(500)
    X_train, X_test, y_train, y_test = train_test_img_split(X, y)

    if model_exists:
        model = tf.keras.models.load_model('twin_model.h5', custom_objects={'DistLayer': DistLayer})
    else:
        model = build_siamese_model()
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics='accuracy')
    es = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True)
    model.fit(
        [X_train[0, :, :, :, :], X_train[1, :, :, :, :]],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_test[0, :, :, :, :], X_test[1, :, :, :, :]], y_test),
        callbacks=[es])
    model.save('twin_model.h5')


if __name__ == '__main__':
    train_model(50, 32)
