# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:55:44 2023

@author: jseys
"""

#https://clemovernet.wordpress.com/2020/01/01/tensorflow-2-prediction-dun-cours-de-bourse-version-simple/

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#####################################   Lire les données
dataframe = pd.read_csv(r".\XAUUSD.csv", names=["Date","TimeStamp", "Open", "High","Low", "Close", "Volume",])
close_serie = dataframe["Close"]
train_values = close_serie.values



#####################################  Normaliser les données
min_max_scaler = MinMaxScaler()
# définir le scaler à partir de l'ensemble des données
scaler = min_max_scaler.fit(train_values.reshape(-1, 1))
# mise à l'échelle des données d'apprentissage et de test
train_values = scaler.transform(close_serie.values[:18000].reshape(-1, 1))
test_values = scaler.transform(close_serie.values[18000:].reshape(-1, 1))


#####################################   Découper en fenetre

def create_window(dataset, start_index, end_index, history_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i])

    return np.array(data), np.array(labels)

train_features, train_labels = create_window(train_values, 0, None, 5)
print(train_features)
print(train_labels)
test_features, test_labels = create_window(test_values, 0, None, 5)

#####################################   parametrage TF

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(100).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(100).repeat()


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(5, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

#####################################   apprentissage

history = model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=180,
    validation_data=test_dataset,
    validation_steps=30
)

#####################################   affichage

def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 100, 0, 0.0018])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

plot_history(history)


































