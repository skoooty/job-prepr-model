import mlflow
import os

def load_model():
    """ Load production model from MLFlow """
    mlflow.set_tracking_uri('https://mlflow.lewagon.ai/')
    mlflow_model_name = '[batch-960]-job_prepr_temp'
    model_uri = f"models:/{mlflow_model_name}/Production"
    try:
        model = mlflow.keras.load_model(model_uri=model_uri)
    except:
        raise Exception(f'No model in Production on mlflow')
    return model

if __name__ == '__main__':
    print(type(load_model()))
