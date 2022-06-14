from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from breeds_loader import BreedsLoader
from helpers.charts import Charts
from helpers.predictor import Predictor
from models.cnn import Cnn
import tensorflow as tf

from models.mobile_net_v2 import MobileNetV2
from models.res_net_50_v2 import ResNet50V2
from tuner import Tuner
import numpy as np


def simple_cnn(loader):
    cnn = Cnn(num_classes=loader.num_breeds, size=loader.img_size)
    cnn.show_summary()
    cnn_results = cnn.fit(loader.train_x, loader.train_y, loader.validation_data)
    cnn.evaluate(loader.test_x, loader.test_y)
    Charts.plot_history(cnn_results.history, title=cnn.model_name, filename=cnn.model_name)


def mobile_net_v2(loader):
    mobile_v2 = MobileNetV2(num_classes=loader.num_breeds, size=loader.img_size)
    mobile_v2.show_summary()
    cnn_results = mobile_v2.fit(loader.train_x, loader.train_y, loader.validation_data)
    mobile_v2.evaluate(loader.test_x, loader.test_y)
    Charts.plot_history(cnn_results.history, title=mobile_v2.model_name, filename=mobile_v2.model_name)


def res_net_50_v2(loader):
    res_v2 = ResNet50V2(num_classes=loader.num_breeds, size=loader.img_size)
    res_v2.show_summary()
    cnn_results = res_v2.fit(loader.train_x, loader.train_y, loader.validation_data)
    res_v2.evaluate(loader.test_x, loader.test_y)
    Charts.plot_history(cnn_results.history, title=res_v2.model_name, filename=res_v2.model_name)


def mobile_net_v2_data_augmentation():
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = data_generator.flow_from_directory(BreedsLoader.sorted_folder_path,
                                                         target_size=(BreedsLoader.img_size, BreedsLoader.img_size),
                                                         batch_size=64,
                                                         subset='training'
                                                         )
    val_generator = data_generator.flow_from_directory(BreedsLoader.sorted_folder_path,
                                                       target_size=(BreedsLoader.img_size, BreedsLoader.img_size),
                                                       batch_size=64,
                                                       subset='validation'
                                                       )

    print(train_generator.class_indices)

    mobile_v2_da = MobileNetV2(num_classes=train_generator.num_classes, size=BreedsLoader.img_size)
    mobile_v2_da.show_summary()
    mobile_v2_da.fit_using_generator(train_generator, val_generator)


def predict_best(model, loader, model_name):
    Predictor.predict("data/custom/x.jpg", model, loader.rev_breed_ids, 'x_' + model_name)
    loss, accuracy = model.evaluate(loader.test_x, loader.test_y)
    print('------------------->', model_name, 'results: ', 'loss:', loss, 'accuracy:', accuracy,
          '<-------------------')
    print('Test accuracy: %.4f%%' % (accuracy * 100))


def predict_custom(model, loader):
    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in loader.test_x]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(loader.test_y, axis=1)) / len(
        predictions)

    results = []
    for i, j in zip(np.array(predictions), np.argmax(loader.test_y, axis=1)):
        predicted_name = loader.rev_breed_ids[i]
        actual_breed = loader.rev_breed_ids[j]
        result = predicted_name + " | " + actual_breed + (
            ' | True' if predicted_name == actual_breed else ' | False ') + str(i) + " " + str(j)
        results.append(result)
    results.sort()
    print(results)

    print('Test accuracy: %.4f%%' % test_accuracy)


def save_as_tf_lite(h5_model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
    tf_lite_model = converter.convert()
    open("saved_models/" + model_name + ".tflite", "wb").write(tf_lite_model)


if __name__ == '__main__':
    # load, prepare and split data
    data_loader = BreedsLoader(verbose=1)
    # BreedsLoader.split_into_folders()

    # simple_cnn(data_loader)
    # predict_best(Cnn.load_saved_model(), data_loader, Cnn.model_name)

    # mobile_net_v2(data_loader)
    # mobile_net_v2_data_augmentation()
    # predict_best(MobileNetV2.load_saved_model(), data_loader, MobileNetV2.model_name)
    # save_as_tf_lite(MobileNetV2.load_saved_model(), MobileNetV2.model_name)
    # tune_mobile_net_v2(data_loader)
    predict_custom(MobileNetV2.load_saved_model(), data_loader)
    # Predictor.predict("data/custom/x.jpg", MobileNetV2.load_saved_model(), data_loader.rev_breed_ids,
    #                   'x_' + MobileNetV2.model_name)

    # res_net_50_v2(data_loader)
    # predict_best(ResNet50V2.load_saved_model(), data_loader, ResNet50V2.model_name)

# ------------------------------------------------------ tuning
# # the best was 16
# def tune_batch_cnn(loader):
#     model = KerasClassifier(build_fn=Cnn.build_cnn_model, num_classes=loader.num_breeds, size=loader.img_size,
#                             verbose=1, epochs=15)
#     batch_size = [16, 32, 64, 128]
#
#     param_grid = dict(batch_size=batch_size)
#     Tuner.tune(model, param_grid, loader.train_x, loader.train_y)
#
#
# # the best was Adamax
# def tune_optimizer_cnn(loader):
#     model = KerasClassifier(build_fn=Cnn.build_cnn_model, num_classes=loader.num_breeds, size=loader.img_size,
#                             verbose=1, epochs=10, batch_size=16)
#     optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam']
#     param_grid = dict(optimizer=optimizer)
#     Tuner.tune(model, param_grid, loader.train_x, loader.train_y)
#
# # the best was 0.0001
# def tune_lr_cnn(loader):
#     model = KerasClassifier(build_fn=Cnn.build_cnn_model, num_classes=loader.num_breeds, size=loader.img_size,
#                             verbose=1, epochs=10, batch_size=16)
#     learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
#     param_grid = dict(learning_rate=learning_rate)
#     Tuner.tune(model, param_grid, loader.train_x, loader.train_y)
#
# def tune_mobile_net_v2(loader):
#     model = KerasClassifier(build_fn=MobileNetV2.build_cnn_model, num_classes=loader.num_breeds, size=loader.img_size,
#                             verbose=1)
#     batch_size = [8, 16, 32, 64, 128]
#     epochs = [8, 14]
#     # optimizer = ['SGD', 'RMSprop', 'Adam', 'Nadam']
#     # activation = ['softmax', 'relu', 'sigmoid']
#     # neurons = [512, 1024]
#     learning_rate = [0.001, 0.0001, 0.00001]
#     # param_grid = dict(neurons=neurons, optimizer=optimizer, activation=activation)
#     # param_grid = dict(learning_rate=learning_rate)
#     param_grid = dict(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
#     Tuner.tune(model, param_grid, loader.train_x, loader.train_y)
