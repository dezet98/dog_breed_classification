import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adamax import Adamax


class Cnn:
    model_name = "cnn"
    model_save_path = "saved_models/" + model_name

    def __init__(self, size, num_classes):
        self.model = Cnn.build_cnn_model(size, num_classes)

    def fit(self, x, y, validation_data):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            # save the Keras model or model weights at some frequency.
            tf.keras.callbacks.ModelCheckpoint(self.model_save_path, verbose=1, save_best_only=True),
        ]

        epochs = 20
        batch_size = 16

        return self.model.fit(x, y,
                              validation_data=validation_data,
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              verbose=1
                              )

    def evaluate(self, test_x, test_y):
        loss, accuracy = self.model.evaluate(test_x, test_y)
        print('------------------->', self.model_name, 'results: ', 'loss:', loss, 'accuracy:', accuracy,
              '<-------------------')

    @staticmethod
    def load_saved_model():
        return tf.keras.models.load_model(Cnn.model_save_path)

    def show_summary(self):
        self.model.summary()

    @staticmethod
    def build_cnn_model(size, num_classes):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="valid",
                                         activation="relu", input_shape=(size, size, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="valid",
                                         activation="relu"))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

        optimizer = Adamax(0.0001)

        # compile model
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

        return model
