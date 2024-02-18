
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta


path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/neural_network/data_processing"
sys.path.append(data_path)
from data_modeling import scale_back_data, train_test_split

EPOCHS=100
PASOS=7

def create_ANNFF_model(PASOS):
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model

def neural_net_FF_model(df):
    x_train, x_val, y_train, y_val = train_test_split(df, 10)
    model = create_ANNFF_model(7)
    model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)
    _, accuracy = model.evaluate(x_val,y_val)
    return model

def labels_pred(last_row):
    features_list = []
    columns = len(last_row)
    for c in range(columns):
        if c != 0:
            sales = last_row[c]
            features_list.append(sales)
    return features_list  

def neural_network_forecast(df_reg, forecast_days, scaler):
    print("Training model")
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    #print('GPU name: ', tf.compat.v1.config.experimental.list_physical_devices('GPU'))
    nn_model = neural_net_FF_model(df_reg)
    df_forecast = df_reg.copy()
    columns = len(df_reg.columns)
    last_row = list(df_reg.values[-1].tolist())
    print('Making predictions')
    for d in range(forecast_days):
        features = labels_pred(last_row)
        features = np.array(features).reshape((1, 1, columns-1))
        prediction = nn_model.predict(features)
        print(prediction[0])
        del last_row[0] 
        last_row.append(prediction[0][0])
        forecast_date = df_forecast.index.max() + timedelta(days=1)
        df_forecast.loc[forecast_date] = last_row 
    df_pred = scale_back_data(df_forecast, scaler)
    df_pred = df_pred[df_pred.columns[-1]]
    return df_pred





