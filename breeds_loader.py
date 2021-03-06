import os

import pandas as pd
from helpers.encoding import Encoding
from helpers.image_loader import ImageLoader
from sklearn.model_selection import train_test_split
import shutil


class BreedsLoader:
    img_size = 224

    def __init__(self, verbose=0):
        self.__df = pd.read_csv('data/labels.csv')
        self.show_head(verbose)

        # extract unique breeds
        self.breeds = self.__df["breed"].unique()
        self.breeds = sorted(self.breeds)
        # number of breads
        self.num_breeds = len(self.breeds)
        self.show_number_of_breeds(verbose)

        # create breed ids set (every breed has own index)
        self.breed_ids = {breed_name: i for i, breed_name in enumerate(self.breeds)}
        self.rev_breed_ids = dict(zip(self.breed_ids.values(), self.breed_ids.keys()))
        self.show_breeds(verbose)

        self.__fetch_and_link_data()
        self.__prepare_data()
        self.__split_data(verbose)

    sorted_folder_path = r"data/sorted"
    train_path = r"data/train/"

    # link images with breeds (create array of images paths and breeds index)
    def __fetch_and_link_data(self):
        images_paths = []
        labels = []
        for image_id, breed_name in zip(self.__df["id"], self.__df["breed"]):
            breed_id = self.breed_ids[breed_name]
            labels.append(breed_id)
            images_paths.append(self.train_path + image_id + ".jpg")

        self.images_paths = images_paths[:1000]
        self.labels = labels[:1000]

    # hot encoding for labels, load images and turn them into vector
    def __prepare_data(self):
        self.y = Encoding.encode(self.labels, self.num_breeds)
        self.x = ImageLoader.load_images(self.images_paths, self.img_size)

    def __split_data(self, verbose=1):
        # split data (ratio 80:20)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=42)
        self.validation_data = (self.test_x, self.test_y)
        self.show_split_data(verbose)

    @staticmethod
    def split_into_folders():
        print("Starting...")
        df = pd.read_csv('data/labels.csv')
        breeds = df["breed"].unique()
        # create breeds folders
        for breed_name in breeds:
            dir_name = BreedsLoader.sorted_folder_path + "/" + breed_name
            if not os.path.exists(BreedsLoader.sorted_folder_path + "/" + breed_name):
                os.mkdir(dir_name)
                print("Directory ", dir_name, " Created ")

        # move images into created folders
        for image_id, breed_name in zip(df["id"], df["breed"]):
            img_name = image_id + ".jpg"
            original_path = BreedsLoader.train_path + img_name
            target_path = BreedsLoader.sorted_folder_path + "/" + breed_name + "/" + img_name
            shutil.copyfile(original_path, target_path)
        print("End...")

    # display methods
    def show_head(self, verbose=1):
        if verbose != 0:
            print(self.__df.head())

    def show_number_of_breeds(self, verbose=1):
        if verbose != 0:
            print("\nNumber of Breeds: ", self.num_breeds)

    def show_breeds(self, verbose=1):
        if verbose != 0:
            print("All breeds:\n", self.breed_ids)

    def show_number_of_train_images(self, verbose=1):
        if verbose != 0:
            print("There is", len(self.__df["id"]), "images in train set")
            print("first id:", self.__df["id"][0])

    def show_dog_breed_id(self, index, verbose=1):
        if verbose != 0:
            if index < len(self.images_paths):
                print("For", index, "image id:", self.images_paths[index], "breed index:", self.labels[index])

    def show_split_data(self, verbose=1):
        if verbose != 0:
            print('train shapes: x -> ', self.train_x.shape, " y -> ", self.train_y.shape, "\ntest shapes: x -> ",
                  self.test_x.shape,
                  " y -> ",
                  self.test_y.shape, sep="")
