"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import os
import bz2
import _pickle as cPickle

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    
    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    daydict = {'Monday' : 1, 'Tuesday' : 1, 'Wednesday' : 1, 'Thursday' : 1, 'Friday' : 1, 'Saturday' : 0, 'Sunday' : 0}
    #data = feature_vector_json
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    #text_file = open('columns_V2.txt', "r")
    #columnlist = text_file.read().splitlines()
    #text_file.close()
    columnlist = ['Barcelona_pressure', 'Barcelona_rain_1h','Barcelona_rain_3h','Barcelona_temp','Barcelona_weather_id',
     'Barcelona_wind_speed', 'Bilbao_clouds_all','Bilbao_pressure','Bilbao_rain_1h','Bilbao_snow_3h','Bilbao_temp',
     'Bilbao_weather_id','Bilbao_wind_speed','Madrid_clouds_all','Madrid_humidity','Madrid_pressure','Madrid_rain_1h',
     'Madrid_temp','Madrid_weather_id','Madrid_wind_speed','Seville_clouds_all','Seville_humidity','Seville_rain_1h',
     'Seville_rain_3h','Seville_temp','Seville_weather_id','Seville_wind_speed','Valencia_humidity','Valencia_pressure',
     'Valencia_snow_3h','Valencia_temp','Valencia_wind_speed','dayofweek','Barcelona_wind_deg_c_North','Barcelona_wind_deg_c_South',
     'Barcelona_wind_deg_c_West','Bilbao_wind_deg_c_North','Bilbao_wind_deg_c_South','Bilbao_wind_deg_c_West','tod_Morning',
     'tod_Night','tod_Noon']
    
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------
    test_alpha = feature_vector_df.copy()
    test_alpha['Barcelona_pressure'] = test_alpha['Barcelona_pressure'].apply(lambda x: 1015 if x >90000 else (1015 if x >1500 or x <900 else x))
    test_alpha['Barcelona_rain_1h'] = test_alpha['Barcelona_rain_1h'].apply(lambda x: 3 if x >3 else x)
    test_alpha['Barcelona_wind_deg_c'] = np.where(test_alpha['Barcelona_wind_deg']<45, 'North',
                  np.where(test_alpha['Barcelona_wind_deg']<135, 'East',
                  np.where(test_alpha['Barcelona_wind_deg']<225, 'South', 'West')))
    test_alpha['Bilbao_wind_deg_c'] = np.where(test_alpha['Bilbao_wind_deg']<45, 'North',
                  np.where(test_alpha['Bilbao_wind_deg']<135, 'East',
                  np.where(test_alpha['Bilbao_wind_deg']<225, 'South', 'West')))
    #fix the max wind speed in valencia
    test_alpha['Valencia_wind_speed'] = test_alpha['Valencia_wind_speed'].apply(lambda x: 13 if x >13 else x)
    test_alpha['Valencia_pressure'] = test_alpha['Valencia_pressure'].fillna(1012)
    #test_alpha['Valencia_pressure'] = test_alpha['Valencia_pressure'].fillna(-99999)
    test_alpha['date'] = pd.to_datetime(test_alpha['time'])
    test_alpha['dayofweek'] = test_alpha['date'].dt.day_name()
    test_alpha["dayofweek"].replace(daydict, inplace=True)
    test_alpha['hour'] = test_alpha['date'].dt.hour
    test_alpha['month'] = test_alpha['date'].dt.month
    test_alpha['tod'] = np.where(test_alpha['hour']<6, 'Night',
                  np.where(test_alpha['hour']<13, 'Morning',
                  np.where(test_alpha['hour']<16, 'Noon',
                  np.where(test_alpha['hour']<22, 'Evening',
                           'Night'))))
    test_alpha['season'] = np.where(test_alpha['month']<4, 'Winter',
                  np.where(test_alpha['month']<7, 'Spring',
                  np.where(test_alpha['month']<10, 'Summer',
                  np.where(test_alpha['month']<13, 'Autumn',
                           'Winter'))))
    test_alpha['bins'] = 1
    #print(len(test_alpha.columns))
    test_alpha.drop(['time','Valencia_wind_deg','Seville_pressure','date','hour','month','bins','Unnamed: 0'], axis=1,inplace=True)
    objlist = ['Barcelona_wind_deg_c','Bilbao_wind_deg_c','tod','season']
    #print(len(test_alpha.columns))
    test_beta = pd.get_dummies(test_alpha,columns = objlist,drop_first=False)
    #test_beta = pd.get_dummies(test_alpha,drop_first=True)
    #print(len(test_beta.columns))
    #len(test_alpha.columns)
    len(columnlist)
    missing = list(set(columnlist) - set(test_alpha.columns.tolist()))
    for item in missing:
        test_beta[item] = np.nan
    
    test_beta = test_beta.fillna(0)
    predict_vector = test_beta[columnlist]

    return predict_vector


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data




def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    #decompress_pickle('smallpickle.pbz2') 
    
    return cPickle.load(bz2.BZ2File(path_to_model, 'rb'))
    #return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
