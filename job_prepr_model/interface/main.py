


def preprocess(source_type='train'):
    pass

def train():
    from job_prepr_model.ml_logic.model import (initialize_model, compile_model, train_model)
    from job_prepr_model.ml_logic.data import load_data
    from job_prepr_model.ml_logic.encoders import label_encode

    X, y = load_data()

    y_cat = label_encode(y)

    #import ipdb; ipdb.set_trace()

    model = initialize_model()
    model = compile_model(model)
    model, history = train_model(model, X, y_cat)

    return model, history


def evaluate():
    pass

def pred(X_pred):
    pass

if __name__ == '__main__':
    #preprocess()
    #preprocess(source_type='val')
    train()
    #pred()
    #evaluate()
