from helpers.charts import Charts
from helpers.predictor import Predictor
from models.cnn import Cnn
import tensorflow as tf

from models.mobile_net_v2 import MobileNetV2
from models.res_net_50_v2 import ResNet50V2


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


def predict_best(model, loader, model_name):
    Predictor.predict("data/custom/x.jpg", model, loader.rev_breed_ids, 'x_' + model_name)
    loss, accuracy = model.evaluate(loader.test_x, loader.test_y)
    print('------------------->', model_name, 'results: ', 'loss:', loss, 'accuracy:', accuracy,
          '<-------------------')
    print('Test accuracy: %.4f%%' % (accuracy * 100))


def save_as_tf_lite(h5_model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
    tf_lite_model = converter.convert()
    open("saved_models/" + model_name + ".tflite", "wb").write(tf_lite_model)


if __name__ == '__main__':
    # load, prepare and split data
    # data_loader = BreedsLoader(verbose=1)

    # simple_cnn(data_loader)
    # predict_best(Cnn.load_saved_model(), data_loader, Cnn.model_name)

    # mobile_net_v2(data_loader)
    # predict_best(MobileNetV2.load_saved_model(), data_loader, MobileNetV2.model_name)
    save_as_tf_lite(MobileNetV2.load_saved_model(), MobileNetV2.model_name)

    # res_net_50_v2(data_loader)
    # predict_best(ResNet50V2.load_saved_model(), data_loader, ResNet50V2.model_name)

# tuning
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
