# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:55:44 2023

@author: jseys
"""


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

WINDOW_SIZE = 50
BATCH_SIZE = 100
EPOCHS = 5
LSTM_SIZE = 100

#resolution des plots
plt.rcParams['figure.dpi'] = 100


#####################################   Lire les données


#CSVdata = pd.read_csv(r".\XAUUSDM5_2004_2023.csv", names=["Date","TimeStamp", "Open", "High","Low", "Close", "Volume",])
CSVdata = pd.read_csv(r".\XAUUSD.csv", names=["Date","TimeStamp", "Open", "High","Low", "Close", "Volume",])

#data = CSVdata[["Open", "High","Low", "Close", "Volume"]]
data = CSVdata[["Open","Close","Volume"]]

close_serie = data["Close"]

pre_norm_data=data.values

# plt.plot(close_serie.values[:],label="data")
# plt.show()

#print ('##### DATA :\n',data)

#####################################  Normaliser les données


min_max_scaler = MinMaxScaler()
# définir le scaler à partir de l'ensemble des données
scaler = min_max_scaler.fit(pre_norm_data.reshape(-1, pre_norm_data.shape[1]))
# mise à l'échelle des données d'apprentissage et de test

norm_data = scaler.transform(data.values[:].reshape(-1, pre_norm_data.shape[1]))
# test_values = scaler.transform(close_serie.values[18000:].reshape(-1, 1))

#print ('##### DATA NORMALISEES :\n',norm_data)

# plt.plot(norm_data[:],label="norm_data")
# plt.show()

#####################################   Découper en fenetre

def create_window(dataset, start_index, end_index, history_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, dataset.shape[1])))
        labels.append(dataset[i,1])

    return np.array(data), np.array(labels)

X, Y = create_window(norm_data, 0, None, WINDOW_SIZE)

# Convertir les données en un format utilisable avec TensorFlow
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


#####################################   parametrage TF

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(BATCH_SIZE).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE).repeat()


model = Sequential([
    layers.LSTM(LSTM_SIZE, input_shape=(WINDOW_SIZE,len(data.columns))),

    layers.Dense(1) ])

model.compile(loss='mse', optimizer='adam')

print(model.summary())

#####################################   apprentissage

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    validation_data=test_dataset,
    validation_steps=len(X_test)//BATCH_SIZE
)

results = model.evaluate(X_test, Y_test, verbose=1)

MODEL_PATH = "./tf_model"
model.save(MODEL_PATH)



#####################################   affichage

def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([0, EPOCHS, 0, 0.000075])#max(max(loss),max(val_loss))])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

plot_history(history)


#####################################   TEST

predictions = model.predict(np.array(X))
#il faut déscaler la predition mais en utilisant que les parametre de scale de la colonne close, du coup on cre un nouveau scaler
close_scaler=MinMaxScaler()
close_scaler.min_,close_scaler.scale_=scaler.min_[1],scaler.scale_[1]

pred_unorm=np.reshape(close_scaler.inverse_transform(predictions), -1)

#pour faire a la main :
#    pred_unorm=(predictions-scaler.min_[1])/scaler.scale_[1]

plt.title("Prix de fermeture")
plt.plot(close_serie.values[(0+WINDOW_SIZE):(50+WINDOW_SIZE)],label='Réél')
plt.grid(True)
plt.plot(pred_unorm[0:50],label='Prédiction')
plt.legend(loc='upper right')
plt.show()


#****** Ecarts
print ("moyenne de écarts ouverture-fermeture :",np.mean(np.abs(data["Close"]-data["Open"])))
print ("moyenne de écarts cloture prédiction-réel :",np.mean(np.abs(pred_unorm[:len(pred_unorm)-1]-np.array(close_serie)[WINDOW_SIZE:len(close_serie)-1])))

plt.plot(pred_unorm[:len(pred_unorm)-1]-np.array(close_serie)[WINDOW_SIZE+1-1:len(close_serie)-1])
plt.show()

# calcul signal
# -1 pour vente et 1 pour achat

pred_variation=pred_unorm-np.array(close_serie)[WINDOW_SIZE-1:len(close_serie)-1]
real_variation= np.array(close_serie)[WINDOW_SIZE:len(close_serie)]-np.array(close_serie)[WINDOW_SIZE-1:len(close_serie)-1]
plt.title("Variation")
plt.plot(real_variation[0:100],label='Réél')
plt.grid(True)
plt.plot(pred_variation[0:100],label='Prédiction')
plt.legend(loc='upper right')
plt.show()
    

    
for i in range(0,(pred_variation.shape[0])):
     if pred_variation[i]>0:
         pred_variation[i]=1
     elif pred_variation[i]<0:
         pred_variation[i]=-1
         
for i in range(0,(real_variation.shape[0])):
     if real_variation[i]>0:
         real_variation[i]=1
     elif real_variation[i]<0:
         real_variation[i]=-1
         
ecart_variation=np.zeros(real_variation.shape)
resultat=0

for i in range(0,(ecart_variation.shape[0])):
     if real_variation[i]==pred_variation[i]:
         ecart_variation[i]=1
         resultat+=1
         
resultat=resultat/ecart_variation.shape[0]*100





















