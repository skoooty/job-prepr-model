import numpy as np
import pandas as pd

from colorama import Fore, Style

#import "Grisearch" params
from job_prepr_model.ml_logic.params import gridsearch_params

def preprocess(source_type='train'):
    pass

learning_rate = 0.001
batch_size = gridsearch_params['batch_size'][-1]
patience = gridsearch_params['earlystopping_patience'][-1]
validation_split = 0.2
epochs=500

maxpooling2d=gridsearch_params['maxpooling2d'][-1]
kernel_size=gridsearch_params['kernel_size'][-1]
kernel_size_detail=gridsearch_params['kernel_size_detail'][-1]
maxpoolinlast_dense_layer_neurons1g2d=gridsearch_params['last_dense_layer_neurons1'][-1]
last_dense_layer_neurons2=gridsearch_params['last_dense_layer_neurons2'][-1]


def train(mode='hd'):
    from job_prepr_model.ml_logic.model import (initialize_model, compile_model, train_model)
    from job_prepr_model.ml_logic.data import (load_train_data, load_train_data_hd, load_validation_data_hd)

    from job_prepr_model.ml_logic.encoders import label_encode
    from job_prepr_model.ml_logic.registry import get_model_version
    from job_prepr_model.ml_logic.registry import load_model, save_model

    y_cat = None
    y=None


    if mode!='hd':
        X, y = load_train_data()
        y_cat_len = label_encode(y)[0].shape[0]
        Xshape = (48, 48, 1)
        y_cat = label_encode(y)
    else:
        X=load_train_data_hd()
        y_cat_len = 8
        Xshape = (100, 100, 1)



    validation_data=load_validation_data_hd()
    #

    #import ipdb; ipdb.set_trace()

    model = initialize_model(X,y_cat_len,Xshape,
                     maxpooling2d=2,
                     activation_for_hidden='relu',
                     kernel_size=(3,3),
                     kernel_size_detail=(2,2),
                     last_dense_layer_neurons_1=100,
                     last_dense_layer_neurons_2=100,
                     )
    model = compile_model(model, learning_rate)

    if mode !='hd':
        model, history = train_model(model, X, y_cat,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs
                                    )
    else:
        model, history = train_model(model, X,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs,
                                    mode='hd',
                                    validation_data=validation_data
                                    )
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
