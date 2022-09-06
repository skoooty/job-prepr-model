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

y_label_dict = {
    'angry':0,
    'disgust':1,
    'fear':2,
    'happy':3,
    'neutral':4,
    'sad':5,
    'surprise':6
}
