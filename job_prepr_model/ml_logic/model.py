from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping



def initialize_model(X,y_cat_len,
                     maxpooling2d=2,
                     activation_for_hidden='relu',
                     kernel_size=(3,3),
                     kernel_size_detail=(2,2),
                     last_dense_layer_neurons_1=100,
                     last_dense_layer_neurons_2=100,
                     ):

    model = models.Sequential()

    # Notice this cool new layer that "pipe" your rescaling within the architecture
    model.add(Rescaling(1./255, input_shape=X[0].shape))

    # Lets add 3 convolution layers, with relatively large kernel size as our pictures are quite big too
    model.add(layers.Conv2D(50, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Conv2D(50, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))
    model.add(layers.MaxPooling2D(maxpooling2d))

    model.add(layers.Conv2D(30, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Conv2D(30, kernel_size=kernel_size, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))
    model.add(layers.MaxPooling2D(maxpooling2d))

    model.add(layers.Conv2D(20, kernel_size=kernel_size_detail, activation=activation_for_hidden))
    model.add(layers.Conv2D(20, kernel_size=kernel_size_detail, activation=activation_for_hidden))
    model.add(layers.Dropout(.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(last_dense_layer_neurons_1, activation='relu'))
    model.add(layers.Dense(last_dense_layer_neurons_2, activation='relu'))
    model.add(layers.Dense(y_cat_len, activation='softmax'))
    return model

def compile_model(model, learning_rate=0.01):
   model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
   return model

def train_model(model,
                X,
                y,
                batch_size=64,
                patience = 2,
                validation_split=0.2,
                epochs=500,
                mode='hd',
                validation_data=None
                ):
    if mode != 'hd':
        history = model.fit(X, y,
                batch_size=batch_size, epochs = epochs,
                callbacks=[EarlyStopping(patience = patience, restore_best_weights= True, monitor = "val_accuracy", mode = "max")],
                validation_split = validation_split, shuffle = True, verbose = 1)
    else:
        model.fit(X, epochs = epochs, validation_data=validation_data,
      callbacks=[EarlyStopping(patience = patience, restore_best_weights= True, monitor = "val_accuracy", mode = "max")], shuffle = True)

    return model, history

def evaluate_model(model, X, y, batch_size=64):
    metrics = model.evaluate(X, y)
    return metrics
