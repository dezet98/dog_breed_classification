import cv2
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


class ImageLoader:

    @staticmethod
    def load_images(img_paths, img_size):
        vectors = [ImageLoader.get_image(img_path, img_size) for img_path in tqdm(img_paths)]

        # normalize from [0, 255] to [0, 1]
        return np.vstack(vectors).astype('float32') / 255
        # vector = np.zeros(dtype='float32', shape=(len(img_paths), img_size, img_size, 3))
        #
        # for index, img_path in enumerate(img_paths):
        #     vector[index] = ImageLoader.get_image(img_path, img_size)
        #
        # return vector

    @staticmethod
    def get_image(path, img_size):
        img = image.load_img(path, target_size=(img_size, img_size))

        # load to shape (img_size, img_size, 3)
        img = image.img_to_array(img)

        # reshape to (1, img_size, img_size, 3)
        return np.expand_dims(img, axis=0)

        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.resize(image, (img_size, img_size))
        #
        # image = preprocess_input(np.expand_dims(np.array(image[..., ::-1].astype(np.float32)).copy(), axis=0))
        # normalize from [0, 255] to [0, 1]
        # image = image.astype(np.float32) / 255
        # return image
