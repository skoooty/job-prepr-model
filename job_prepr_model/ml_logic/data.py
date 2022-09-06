import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
from job_prepr_model.ml_logic.params import LOCAL_DATA_PATH



train_path = os.path.join(LOCAL_DATA_PATH, 'train')
test_path = "/Users/andrei/code/images/images/validation"
val_path = "~/code/images/images/val"

def load_data():

    #all_images = []
    y_train = []
    X_train = []
    for folder_path in os.listdir(train_path):
        if not folder_path.startswith("."):
            for image_path in os.listdir(os.path.join(train_path, folder_path)):
                img = load_img(os.path.join(os.path.join(train_path, folder_path), image_path), color_mode = "grayscale")
                X_train.append(img_to_array(img))
                y_train.append(os.path.basename(os.path.normpath(folder_path)))
    return np.array(X_train), np.array(y_train)
