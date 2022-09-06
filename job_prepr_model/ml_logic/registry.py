import mlflow
import os
from colorama import Fore, Style

def load_model():
    """ Load production model from MLFlow """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_uri = f"models:/{mlflow_model_name}/Production"
    try:
        model = mlflow.keras.load_model(model_uri=model_uri)
    except:
        raise Exception(f'No model in Production on mlflow at URI {os.environ.get("MLFLOW_TRACKING_URI")}')
    return model
