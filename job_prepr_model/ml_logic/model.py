from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping



def initialize_model():

    model = models.Sequential()

    # Notice this cool new layer that "pipe" your rescaling within the architecture
    model.add(Rescaling(1./255, input_shape=(48, 48, 1)))

    # Lets add 3 convolution layers, with relatively large kernel size as our pictures are quite big too
    model.add(layers.Conv2D(16, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling2D(3))

    model.add(layers.Conv2D(32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    model.add(layers.Conv2D(32, kernel_size=2, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
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
                validation_split=0.2
                ):
    history = model.fit(X, y,
              batch_size=batch_size, epochs = 1000,
              callbacks=[EarlyStopping(patience = patience, restore_best_weights= True, monitor = "val_accuracy", mode = "max")],
              validation_split = validation_split, verbose = 1)

    return model, history

def evaluate_model(model, X, y, batch_size=64):
    metrics = model.evaluate(X, y)
    return metrics
