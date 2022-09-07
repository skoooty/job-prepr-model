import numpy as np
import pandas as pd

from colorama import Fore, Style


def preprocess(source_type='train'):
    pass

learning_rate = 0.001
batch_size = 256
patience = 2
validation_split = 0.2
epochs=10

def train():
    from job_prepr_model.ml_logic.model import (initialize_model, compile_model, train_model)
    from job_prepr_model.ml_logic.data import load_train_data
    from job_prepr_model.ml_logic.encoders import label_encode
    from job_prepr_model.ml_logic.registry import get_model_version
    from job_prepr_model.ml_logic.registry import load_model, save_model

    X, y = load_train_data()

    y_cat = label_encode(y)

    #import ipdb; ipdb.set_trace()

    model = initialize_model()
    model = compile_model(model, learning_rate)
    model, history = train_model(model, X, y_cat,
                                  batch_size=batch_size,
                                  validation_split=validation_split,
                                  epochs=epochs)

    params = dict(
        # model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        # package behavior
        context="train",
        #chunk_size=CHUNK_SIZE,
        # data source
        validation_split = validation_split,
        #row_count=row_count,
        model_version=get_model_version()
        #dataset_timestamp=get_dataset_timestamp(),
    )
    val_accuracy = np.min(history.history['val_accuracy'])

    save_model(model=model, params=params, metrics=dict(mae=val_accuracy))

    return val_accuracy


def validate():
    from job_prepr_model.ml_logic.model import evaluate_model
    from job_prepr_model.ml_logic.data import load_test_data
    from job_prepr_model.ml_logic.encoders import label_encode
    from job_prepr_model.ml_logic.registry import load_model, save_model
    from job_prepr_model.ml_logic.registry import get_model_version

    # load new data
    X, y = load_test_data()

    y_cat = label_encode(y)

    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X, y=y_cat)

    #import ipdb; ipdb.set_trace()

    accuracy = metrics_dict[1]

    # save evaluation
    params = dict(
        # model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        # package behavior
        context="train",
        #chunk_size=CHUNK_SIZE,
        # data source
        validation_split = validation_split,
        #row_count=row_count,
        model_version=get_model_version()
        #dataset_timestamp=get_dataset_timestamp(),
    )

    save_model(params=params, metrics=dict(accuracy=accuracy))

    return accuracy

def pred(X_pred):
    pass

if __name__ == '__main__':
    #preprocess()
    #preprocess(source_type='val')
    train()
    validate()
    #pred()
    #evaluate()
