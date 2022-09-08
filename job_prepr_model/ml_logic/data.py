import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
from job_prepr_model.ml_logic.params import LOCAL_DATA_PATH
from job_prepr_model.ml_logic.params import LOCAL_DATA_PATH_HD


#Import for HD data
#from keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.keras.preprocessing import image_dataset_from_directory
#import tensorflow_datasets as tfds
import numpy as np


train_path = os.path.join(LOCAL_DATA_PATH, 'train')
test_path = os.path.join(LOCAL_DATA_PATH, 'validation')
hd_path = os.path.join(LOCAL_DATA_PATH_HD,'archive')
#val_path = "~/code/images/images/val"

def load_data(data_path):

    #all_images = []
    y_train = []
    X_train = []
    for folder_path in os.listdir(data_path):
        if not folder_path.startswith("."):
            for image_path in os.listdir(os.path.join(data_path, folder_path)):
                img = load_img(os.path.join(os.path.join(data_path, folder_path), image_path), color_mode = "grayscale")
                X_train.append(img_to_array(img))
                y_train.append(os.path.basename(os.path.normpath(folder_path)))
    return np.array(X_train), np.array(y_train)


def load_train_data():

    return load_data(train_path)

def load_test_data():

    return load_data(test_path)

def load_train_data_hd(sample=None):
    training_data = image_dataset_from_directory(
        hd_path,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(100, 100),
        validation_split=0.2,
        subset='training',
        seed=0,
        batch_size=64
    )
    if sample:
        ds_len = int((training_data.cardinality().numpy()*(1-sample)))
        data = training_data.skip(ds_len)
    else:
        data = training_data

    return data

def load_validation_data_hd(sample=None):
    validation_data = image_dataset_from_directory(
        hd_path,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(100, 100),
        validation_split=0.2,
        subset='validation',
        seed=0,
        batch_size=64
    )
    if sample:
        ds_len = int((validation_data.cardinality().numpy()*(1-sample)))
        data = validation_data.skip(ds_len)
    else:
        data = validation_data

    return data
