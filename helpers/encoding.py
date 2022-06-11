import numpy as np


class Encoding:

    # hot encoding
    @staticmethod
    def encode(x, num_classes):
        vector = np.zeros(dtype='int32', shape=(len(x), num_classes))
        for index, value in enumerate(x):
            vector[index][value] = 1
        return vector
