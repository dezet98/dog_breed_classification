import pandas as pd
from glob import glob
from helpers.encoding import Encoding
from helpers.image_loader import ImageLoader


class BreedsLoader:
    img_size = 224

    def __init__(self):
        self.__df = pd.read_csv('data/labels.csv')

        # extract unique breeds
        self.breeds = self.__df["breed"].unique()
        # number of breads
        self.num_breeds = len(self.breeds)

        # create breed ids set (every breed has own index)
        self.breed_ids = {breed_name: i for i, breed_name in enumerate(self.breeds)}
        self.rev_breed_ids = dict(zip(self.breed_ids.values(), self.breed_ids.keys()))

        self.__fetch_and_link_data()
        self.__prepare_data()

    # link images with breeds (create array of images paths and breeds index)
    def __fetch_and_link_data(self):
        train_path = r'data\train\*'
        images_paths = glob(train_path)
        ids = list(map(lambda image_path: image_path.split(".")[0].replace(train_path[:-1], ''), images_paths))
        print("There is", len(ids), "images in train set")
        print("first id:", ids[0])

        labels = []
        for image_id in ids:
            breed_name = list(self.__df[self.__df.id == image_id]["breed"])[0]
            breed_idx = self.breed_ids[breed_name]
            labels.append(breed_idx)

        self.__images_paths = images_paths[:1000]
        self.__labels = labels[:1000]
        print("-----Done-----")
        print("Example:", "image id:", images_paths[0], "breed index:", labels[0])

    # hot encoding for labels, load images
    def __prepare_data(self):
        self.y = Encoding.encode(self.__labels, self.num_breeds)
        self.x = ImageLoader.load_images(self.__images_paths, self.img_size)

    # display methods
    def show_head(self):
        print(self.__df.head())

    def show_number_of_breeds(self):
        print("\nNumber of Breeds: ", self.num_breeds)

    def show_breeds(self):
        print("All breeds:\n", self.breed_ids)
