
# # **import packages**


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn import metrics
import keras
import tensorflow
from tensorflow.keras.layers import LSTM,Bidirectional
import time
import os


# # **read data**

input_data=pd.read_csv("speed.csv",header=0)
data=input_data
input_data.head()

data.shape


# # **drop unnecessary columns**


data.head()


# # **preprocess data**


data_np=np.array(data)
scaler = MinMaxScaler()
scaler.fit(data_np)
all_data_normalized=scaler.transform(data_np)

#drop label
lbl_normalized=all_data_normalized[:,-1]
data_normalized=all_data_normalized

print("label shape",lbl_normalized.shape)
print("data shape",data_normalized.shape)


data_normalized


my_epoch=30
my_batch_size=32


# # **create  time series data**


window_size= 16
nfeature=data_normalized.shape[1]

X=[]
y=[]
for i in range(len(data_normalized)-window_size):
    t=[]
    for j in range(0,window_size):
        t.append(data_normalized[[(i+j)], :])
    X.append(t)
    y.append(lbl_normalized[i+ window_size])


data_time_window, lbl_time_window= np.array(X), np.array(y)
data_time_window= data_time_window.reshape(data_time_window.shape[0],window_size, nfeature)
print(data_time_window.shape)
print(lbl_time_window.shape)



data_time_window[1:16]



lbl_time_window[1:16]


# # **split data into train and test sets**


#split data to train and test sets
train_idx= round(.8 * (data_time_window.shape[0]))
train_data=data_time_window[:train_idx,:]
train_lbl=lbl_time_window[:train_idx]
test_data=data_time_window[train_idx:,:]
test_lbl=lbl_time_window[train_idx:]


# # **GRU Model**


GRU_model = keras.models.Sequential()
GRU_model.add(keras.layers.GRU(units=100,kernel_initializer='glorot_uniform',  input_shape=(data_time_window.shape[1],nfeature)))
GRU_model.add(keras.layers.Dense(units=1 ))
GRU_model.compile(optimizer="Adam",loss=keras.losses.mean_squared_error,metrics=[tensorflow.metrics.MeanAbsoluteError()])
GRU_model.summary()

# # **Train GRU Model**


# GRU_model_hist=GRU_model.fit(train_data,train_lbl,epochs=my_epoch,batch_size=my_batch_size)

start_time = time.perf_counter()

GRU_model_hist = GRU_model.fit(
    train_data,
    train_lbl,
    epochs=my_epoch,
    batch_size=my_batch_size
)

end_time = time.perf_counter()

training_time_seconds = end_time - start_time

print(f"Training time: {training_time_seconds:.2f} seconds")
print(f"Training time: {training_time_seconds/60:.2f} minutes")


# # **plot loss and mae of GRU during train**

MAE = GRU_model_hist.history['mean_absolute_error']
loss = GRU_model_hist.history['loss']

plt.figure(figsize=(10,8))
plt.plot(range(my_epoch), loss, label='training loss')
plt.legend(loc='upper right')
plt.title('Training loss ')
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.show()
plt.figure(figsize=(10,8))
plt.plot(range(my_epoch), MAE, label='training MAE' )
plt.legend(loc='upper right')
plt.title('Training MAE')
#plt.show()


# # **test GRU model**


y_pred_GRU=GRU_model.predict(test_data,verbose=2)
train_pred_GRU=GRU_model.predict(train_data,verbose=2)
mse=metrics.mean_squared_error(test_lbl, y_pred_GRU)
rmse=metrics.mean_squared_error(test_lbl, y_pred_GRU)**0.5
mae= metrics.mean_absolute_error(test_lbl,y_pred_GRU)
smape = 100 * np.mean(2 * np.abs(y_pred_GRU - test_lbl) / (np.abs(test_lbl) + np.abs(y_pred_GRU) + 1e-8))
print('Mean squared error (MSE): %.3f ' % mse)
print('Root mean square error (RMSE) : %.3f '%rmse)
print('Mean Absolute Error (MAE): %.3f'%mae)
print("R2 Score",r2_score(test_lbl,y_pred_GRU ))
print('sMAPE: %.3f %%' % smape)

