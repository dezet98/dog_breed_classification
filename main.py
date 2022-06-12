from breeds_loader import BreedsLoader
from helpers.charts import Charts
from helpers.predictor import Predictor
from models.cnn import Cnn
import numpy as np


def simple_cnn(loader):
    cnn = Cnn(num_classes=loader.num_breeds, size=loader.img_size)
    cnn.show_summary()
    cnn_results = cnn.fit(loader.train_x, loader.train_y, loader.validation_data)
    cnn.evaluate(loader.test_x, loader.test_y)
    Charts.plot_history(cnn_results.history, title=cnn.model_name, filename=cnn.model_name)


def predict_best(model, loader):
    Predictor.predict("data/custom/boxer.jpg", model, loader.rev_breed_ids)
    loss, accuracy = model.evaluate(loader.test_x, loader.test_y)
    print('------------------->', "cnn", 'results: ', 'loss:', loss, 'accuracy:', accuracy,
          '<-------------------')

    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in loader.test_x]
    test_accuracy = 100 * np.sum(np.array(dog_breed_predictions) == np.argmax(loader.test_y, axis=1)) / len(
        dog_breed_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


if __name__ == '__main__':
    # load, prepare and split data
    data_loader = BreedsLoader(verbose=0)

    simple_cnn(data_loader)
    predict_best(Cnn.load_saved_model(), data_loader)

    # tune_lr_cnn(data_loader)

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
