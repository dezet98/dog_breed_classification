from helpers.charts import Charts
from helpers.image_loader import ImageLoader
import numpy as np


class Predictor:
    @staticmethod
    def predict(image_path, model, rev_breed_ids):
        image = ImageLoader.get_image(image_path, 224)

        prediction = model.predict(image)[0]
        label_id = np.argmax(prediction)

        breed_name = rev_breed_ids[label_id]
        Charts.plot_dog_prediction(image_path, breed_name, breed_name)
