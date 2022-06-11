import cv2
import numpy as np


class ImageLoader:

    @staticmethod
    def load_images(img_paths, img_size):
        vector = np.zeros(dtype='float32', shape=(len(img_paths), img_size, img_size, 3))

        for index, img_path in enumerate(img_paths):
            vector[index] = ImageLoader.get_image(img_path, img_size)

        return vector

    @staticmethod
    def get_image(path, img_size):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_size, img_size))

        # normalize from [0, 255] to [0, 1]
        image = image.astype(np.float32) / 255

        return image
