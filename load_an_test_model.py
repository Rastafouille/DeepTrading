# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:06:26 2023

@author: jseys
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#resolution des plots
plt.rcParams['figure.dpi'] = 150

MODEL_PATH = "./tf_model"
model = tf.keras.models.load_model(MODEL_PATH)




#####################################   Lire les données

#CSVdata = pd.read_csv(r".\XAUUSDM5_2004_2023.csv", names=["Date","TimeStamp", "Open", "High","Low", "Close", "Volume",])
CSVdata = pd.read_csv(r".\XAUUSD.csv", names=["Date","TimeStamp", "Open", "High","Low", "Close", "Volume",])

#data = CSVdata[["Open", "High","Low", "Close", "Volume"]]
data = CSVdata[["Open","Close","Volume"]]

close_serie = data["Close"]

pre_norm_data=data.values















predictions = model.predict(np.array(X))
#il faut déscaler la predition mais en utilisant que les parametre de scale de la colonne close, du coup on cre un nouveau scaler
close_scaler=MinMaxScaler()
close_scaler.min_,close_scaler.scale_=scaler.min_[1],scaler.scale_[1]

pred_unorm=np.reshape(close_scaler.inverse_transform(predictions), -1)

#pour faire la main :
#    pred_unorm=(predictions-scaler.min_[1])/scaler.scale_[1]


plt.plot(close_serie.values[(100+WINDOW_SIZE):(150+WINDOW_SIZE)])
plt.grid(True)
plt.plot(pred_unorm[100+1:150+1])
plt.show()


#****** Ecarts
print ("moyenne de écarts ouverture-fermeture :",np.mean(np.abs(data["Close"]-data["Open"])))
print ("moyenne de écarts cloture prédiction-réel :",np.mean(np.abs(pred_unorm[:len(pred_unorm)-1]-np.array(close_serie)[WINDOW_SIZE+1:])))

plt.plot(pred_unorm[:len(pred_unorm)-1]-np.array(close_serie)[WINDOW_SIZE+1-1:len(close_serie)-1])
plt.show()

# calcul signal
# -1 pour vente et 1 pour achat

pred_variation=pred_unorm-np.array(close_serie)[WINDOW_SIZE-1:len(close_serie)-1]
for i in range(0,(pred_variation.shape[0])):
     if pred_variation[i]>0:
         pred_variation[i]=1
     elif pred_variation[i]<0:
         pred_variation[i]=-1
         
         
real_variation= np.array(close_serie)[WINDOW_SIZE:len(close_serie)]-np.array(close_serie)[WINDOW_SIZE-1:len(close_serie)-1]
        
for i in range(0,(real_variation.shape[0])):
     if real_variation[i]>0:
         real_variation[i]=1
     elif real_variation[i]<0:
         real_variation[i]=1
         
ecart_variation=np.zeros(real_variation.shape)
resultat=0

for i in range(0,(ecart_variation.shape[0])):
     if real_variation[i]==pred_variation[i]:
         ecart_variation[i]=1
         resultat+=1
         
resultat=resultat/ecart_variation.shape[0]*100