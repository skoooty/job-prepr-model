import os
import numpy as np

LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))

MLFLOW_TRACKING_URI=os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT=os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME=os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_BACKEND=os.environ.get("PREFECT_BACKEND")
PREFECT_FLOW_NAME=os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL=os.environ.get("PREFECT_LOG_LEVEL")
PREFECT_PROJECT_NAME=os.environ.get("PREFECT_PROJECT_NAME")

gridsearch_params = {
    # total number of combinations: 1296
    optimizer : ["adam"],
    maxpooling2d : [2, 3, 4],
    activation_for_hidden : ["relu", "tanh"],
    kernel_size : [(3, 3)],
    last_dense_layer_neurons : [80, 90, 100, 110, 120],
    batch_size : [8, 16, 32, 64],
    earlystopping_patience : [50]
}

y_label_dict = {
    'angry':0,
    'disgust':1,
    'fear':2,
    'happy':3,
    'neutral':4,
    'sad':5,
    'surprise':6
}
