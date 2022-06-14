
import tensorflow as tf


class ResNet50V2:
    model_name = "ResNet50V2"
    model_save_path = "saved_models/" + model_name + ".h5"

    def __init__(self, size, num_classes):
        self.model = ResNet50V2.build_cnn_model(size, num_classes)

    def fit(self, x, y, validation_data):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
            # save the Keras model or model weights at some frequency.
            tf.keras.callbacks.ModelCheckpoint(self.model_save_path, verbose=1, save_best_only=True),
        ]

        epochs = 30
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
        return tf.keras.models.load_model(ResNet50V2.model_save_path)

    def show_summary(self):
        self.model.summary()

    @staticmethod
    def build_cnn_model(size, num_classes):
        # shape of image
        inputs = tf.keras.Input((size, size, 3))

        # building model
        res_net50_v2 = tf.keras.applications.ResNet50V2(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet")

        # freeze trainable layers
        res_net50_v2.trainable = False

        # add  global average pooling
        x = res_net50_v2.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # add fully connected layer
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        # and output layer
        x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, x)

        # compile model
        alpha = 0.0001
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(alpha),
                      metrics=["acc"])

        return model
