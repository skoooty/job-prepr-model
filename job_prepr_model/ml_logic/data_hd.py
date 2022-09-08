"""
Load data from bucket - to be able to train VM from bucket data directly
Doesn't work for keras image_dataset_from_directory - so not worth to continue
"""

import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
from job_prepr_model.ml_logic.params import LOCAL_DATA_PATH

import logging
import os
from google.cloud import storage

#import webapp2

#from google.appengine.api import app_identity

# train_path = os.path.join(LOCAL_DATA_PATH, 'train')
# test_path = os.path.join(LOCAL_DATA_PATH, 'validation')
# #val_path = "~/code/images/images/val"

def get(self):
  bucket_name = os.environ.get('BUCKET_NAME',
                               app_identity.get_default_gcs_bucket_name())

  self.response.headers['Content-Type'] = 'text/plain'
  self.response.write('Demo GCS Application running from Version: '
                      + os.environ['CURRENT_VERSION_ID'] + '\n')
  self.response.write('Using bucket name: ' + bucket_name + '\n\n')

def read_file(self, filename):
    self.response.write('Reading the full file contents:\n')

    gcs_file = gcs.open(filename)
    contents = gcs_file.read()
    gcs_file.close()
    self.response.write(contents)

def list_bucket(self, bucket):
  """Create several files and paginate through them.

  Production apps should set page_size to a practical value.

  Args:
    bucket: bucket.
  """
  self.response.write('Listbucket result:\n')

  page_size = 1
  stats = gcs.listbucket(bucket + '/foo', max_keys=page_size)
  while True:
    count = 0
    for stat in stats:
      count += 1
      self.response.write(repr(stat))
      self.response.write('\n')

    if count != page_size or count == 0:
      break
    stats = gcs.listbucket(bucket + '/foo', max_keys=page_size,
                           marker=stat.filename)


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


# def load_train_data():

#     return load_data(train_path)

# def load_test_data():

#     return load_data(test_path)
