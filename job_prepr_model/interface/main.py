import numpy as np
import pandas as pd

from colorama import Fore, Style

#import "Grisearch" params
from job_prepr_model.ml_logic.params import (gridsearch_params,batch_size, learning_rate)
from job_prepr_model.utils.gridsearch import gridsearch_params_list
def preprocess(source_type='train'):
    pass


#batch_size = 128 #gridsearch_params['batch_size'][-1]
patience = 20 #gridsearch_params['earlystopping_patience'][-1]
validation_split = 0.2
epochs=500



maxpooling2d=gridsearch_params['maxpooling2d'][-1]
kernel_size= gridsearch_params['kernel_size'][-1]
kernel_size_detail=gridsearch_params['kernel_size_detail'][-1]
last_dense_layer_neurons1=gridsearch_params['last_dense_layer_neurons1'][-1]
last_dense_layer_neurons2=gridsearch_params['last_dense_layer_neurons2'][-1]
activation_for_hidden=gridsearch_params['activation_for_hidden'][-1]





def train(mode='hd', sample=None):
    from job_prepr_model.ml_logic.model import (initialize_model, compile_model, train_model)
    from job_prepr_model.ml_logic.data import (load_train_data, load_train_data_hd, load_validation_data_hd)

    from job_prepr_model.ml_logic.encoders import label_encode
    from job_prepr_model.ml_logic.registry import get_model_version
    from job_prepr_model.ml_logic.registry import load_model, save_model

    dataset_sample = sample

    y_cat = None
    y=None



    if mode!='hd':
        X, y = load_train_data()
        y_cat_len = label_encode(y)[0].shape[0]
        Xshape = (48, 48, 1)
        y_cat = label_encode(y)
    else:
        X=load_train_data_hd(sample=dataset_sample)
        y_cat_len = 8
        Xshape = (100, 100, 1)



    validation_data=load_validation_data_hd(sample=dataset_sample)




    model = initialize_model(X,y_cat_len,Xshape,
                     maxpooling2d=maxpooling2d,
                     activation_for_hidden=activation_for_hidden,
                     kernel_size=kernel_size,
                     kernel_size_detail=kernel_size_detail,
                     last_dense_layer_neurons_1=last_dense_layer_neurons1,
                     last_dense_layer_neurons_2=last_dense_layer_neurons2,
                     )
    model = compile_model(model, learning_rate=None)

    #import ipdb; ipdb.set_trace()

    if mode !='hd':
        model, history = train_model(model, X, y_cat,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs
                                    )
    else:
        model, history = train_model(model, X,
                                    batch_size=batch_size,
                                    #validation_split=validation_split,
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

    return history


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


def gridsearch_model(sample=None, epochs=epochs):

    gp_list = gridsearch_params_list()
    results = {
        'grid_params':[],
        'history':[],
        'val_accuracy':[]
    }
    for params_dict in gp_list:

        maxpooling2d=params_dict['maxpooling2d']
        kernel_size=params_dict['kernel_size']
        kernel_size_detail=params_dict['kernel_size_detail']
        last_dense_layer_neurons1=params_dict['last_dense_layer_neurons1']
        activation_for_hidden=params_dict['activation_for_hidden']

        results['grid_params'].append(params_dict)
        history = train(mode='hd',sample=sample)
        results['history'].append(history)
        val_accuracy = validate()
        results['val_accuracy'].append(val_accuracy)

    return results



if __name__ == '__main__':
    #preprocess()
    #preprocess(source_type='val')
    train()
    validate()
    #pred()
    #evaluate()
