# Standard Libraries
import os
import random
from pathlib import Path

# Third Party Imports
import numpy as np
import progressbar
from matplotlib import image as img
from skimage.transform import resize
from sklearn.model_selection import train_test_split



def load_raw_covid_data(limit):
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    covid_path = os.path.join(script_dir, "data", "covid")
    non_covid_path = os.path.join(script_dir, "data", "noncovid")
    covid_images = [f for f in os.listdir(covid_path) if f.endswith('.png')]
    non_covid_images = [f for f in os.listdir(non_covid_path) if f.endswith('.png')]

    # Randomly samples "limit" amount of images for the train-test phase
    if limit != None:
        covid_images = random.sample(covid_images, limit)
        non_covid_images = random.sample(non_covid_images, limit)

    IMG_SIZE = 128

    # Two empty numpy arrays to store coverted images
    positive_npy = np.empty(
        (len(covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    negative_npy = np.empty(
        (len(non_covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    # start a bar of show percentage of loading data
    covid_bar = progressbar.ProgressBar(maxval=len(covid_images), widgets=[
                                        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    non_covid_bar = progressbar.ProgressBar(maxval=len(non_covid_images), widgets=[
                                            progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    covid_bar.start()
    # Converting COVID dataset to .npy format
    for i, _file in enumerate(covid_images):
        try:
            image_npy = img.imread(_file)
            positive = resize(
                image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
            positive_npy[i] = positive
        except:
            pass
        covid_bar.update(i+1)

    covid_bar.finish()
    print("COVID images converting done")

    non_covid_bar.start()
    # Converting non-COVID dataset to .npy format
    for i, _file in enumerate(non_covid_images):
        try:
            image_npy = img.imread(_file)
            negative = resize(
                image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
            negative_npy[i] = negative
        except:
            pass
        non_covid_bar.update(i+1)
    non_covid_bar.finish()
    print("non covid images converting done")

    positive = positive_npy
    positive_labels = ["1" for i in positive]
    negative = negative_npy
    negative_labels = ["0" for i in negative]

    # Joining both datasets and labels
    X = np.concatenate([positive, negative])
    y = np.array((positive_labels + negative_labels), dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
