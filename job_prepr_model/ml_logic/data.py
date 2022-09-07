import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
from job_prepr_model.ml_logic.params import LOCAL_DATA_PATH

train_path = os.path.join(LOCAL_DATA_PATH, 'train')
test_path = os.path.join(LOCAL_DATA_PATH, 'validation')
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
