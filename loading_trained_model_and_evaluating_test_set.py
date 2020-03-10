#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:08:40 2020

@author: Ivomar Brito Soares
"""
import pandas as pd
from keras.models import model_from_json
from sklearn.metrics import classification_report

# TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing_and_training import basic_preprocessing
from preprocessing_and_training import prepare_targets


def loading_model_from_files(model_json_file, model_h5_file):
    """Loading saved keras model from JSON file and weights from HDF5 file."""
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5_file)
    print("Loaded model from disk")
    return loaded_model


if __name__ == "__main__":

    # Defining variables
    path_to_evaluation_dataset = 'test_data.csv'
    categorical_target_name = 'categorical_target_1'
    features_column_name = 'features'
    nb_classes = 43      # Chosen target variable, categorical_target_1 with 43 unique values or classes.
    model_json_file = 'model.json'
    model_h5_file = 'model.h5'
    batch_size = 64
    
    # Reading dataset
    dataset = pd.read_csv(path_to_evaluation_dataset)
    
    # Preprocessing
    dataset.dropna(subset=[categorical_target_name], inplace=True)
    basic_preprocessing(dataset, features_column_name)
    
    # Feature Extraction: Term Frequency - Inverse Document Frequency (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', max_features= 10000,strip_accents='unicode', norm='l2')
    X_test = tfidf_vectorizer.fit_transform(dataset[features_column_name]).todense()
    
    # Preparing categorical target variable
    y_test_enc = prepare_targets(dataset[categorical_target_name])
    
    # Loading trained deep learning model
    loaded_model = loading_model_from_files(model_json_file, model_h5_file)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Model evaluation and performance report
    y_test_predclass = loaded_model.predict_classes(X_test,batch_size=batch_size)
    print ("Deep Neural Network - Test Classification Report")
    print (classification_report(y_test_enc,y_test_predclass))