import os
import keras
import pickle
import numpy as np
from .data_loader import intents_list, word_tokenize, stemmer, itertools

dir_path = os.path.dirname(os.path.realpath(__file__))
training, exit_data, tags, all_words = [], [], [], []

model_path = os.path.join(dir_path, "..", "EntrenamientoPickle", "brain_model.h5")

# Intentar cargar modelo entrenado
if os.path.isfile(model_path):
    model = keras.models.load_model(model_path)
else:
    # Reentrenar si no existe
    training = np.array(training)
    exit_data = np.array(exit_data)
    model = keras.Sequential([
        keras.layers.Input(shape=(len(training[0]),)),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(exit_data[0]), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(training, exit_data, epochs=2000, batch_size=128, validation_split=0.1, verbose=1)
    model.save(model_path)
