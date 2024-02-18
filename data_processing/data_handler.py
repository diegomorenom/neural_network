import pandas as pd

import os
import datetime 

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_base_path = str(parent_path)+"/neural_network/data_processing/data_base"
forecast_path = str(parent_path)+"/neural_network/data_processing/forecast_files"



def get_data():
    data = pd.read_csv(data_base_path+"/train.csv")  
    return data

def get_splitted_df(data, family, store_nbr):
    df_info = data[(data['family']==family)&(data['store_nbr']==store_nbr)]
    df_info = pd.pivot_table(df_info, values='sales', index=['store_nbr', 'family','date'], aggfunc="sum").reset_index()
    df_info = df_info[['date', 'sales']]
    return df_info

def get_time_series(df_info):
    df_info['date'] = pd.to_datetime(df_info['date'])
    date_range = pd.date_range(df_info['date'].min(),df_info['date'].max(),freq='d')
    df_ts = pd.DataFrame({'date':date_range})
    df_ts = df_ts.merge(df_info, how='left', on='date')
    df_ts = df_ts.set_index('date')
    df_ts.index = pd.DatetimeIndex(df_ts.index).to_period('D')
    return df_ts

def fill_values(df_ts):
    df_ts = df_ts.fillna(0)
    return df_ts

def get_stores(data):
    stores = list(data['store_nbr'].unique())
    return stores

def get_families(data):
    families = list(data['family'].unique())
    return families

def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

def structure_predictions(date, df_pred, family, store_nbr):
    df_pred = pd.DataFrame(df_pred).reset_index()#, columns=['forecast_date','forecast'])
    df_pred.columns = ['forecast_date','forecast']
    df_pred['date'] = date
    df_pred['family'] = family
    df_pred['store_nbr'] = store_nbr
    df_pred['date_updated'] = datetime.datetime.now()
    return df_pred
    
def save_predictions(date, df_pred):
    print('Saving predictions')
    file_name = forecast_path+'/forecast_newralnetwork_'+str(date.replace('-', ''))+'.csv'
    df_pred['forecast_date'] = df_pred['forecast_date'].astype(str)
    df_pred['forecast_date'] = pd.to_datetime(df_pred['forecast_date'])
    df_pred = df_pred.loc[df_pred['forecast_date'] > date]
    df_pred.to_csv(file_name, mode='a', index=False, header=True)
    return "Forecast saved"



 


#values = df.values

#values = values.astype('float32')

#scaler = MinMaxScaler(feature_range=(-1, 1))
#values=values.reshape(-1, 1) 
#scaled = scaler.fit_transform(values)

#reframed = series_to_supervised(scaled, PASOS, 1)
#reframed.head()