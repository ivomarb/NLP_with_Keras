#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:29:49 2020

@author: Ivomar Brito Soares
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Data pre-processing modules
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from textblob import Word
from sklearn import preprocessing

# TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep Learning modules
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

BATCH_SIZE = 64

def basic_preprocessing(dataset, feature_name):
    """
    These are the basic pre-processing steps followed in this function:
    - Convert text to lower case.
    - Punctuation removal.
    - Stop words removal.
    
    Additional possible pre-processing steps (future work):
    - Common words removal.
    - Rare words removal.
    - Spelling correction.
    - Keeping words of length of at least 3.
    """   
    # The first pre-processing is to convert all text into lower case, this avoids having multiple copies
    # of the same words.
    dataset[feature_name] = dataset[feature_name].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Punctuation removal, often it does not add extra information when dealing with text data. Removing them helps
    # reduce the size of the training data.
    dataset[feature_name] = dataset[feature_name].str.replace('[^\w\s]','')
    
    # Stop words (frequently occurring words) should be removed from the dataset.
    stop = stopwords.words('english')
    dataset[feature_name] = dataset[feature_name].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    # Lemmatization: Converts the word into its root word.
    dataset[feature_name] = dataset[feature_name].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    

def prepare_targets(y_train):
    """
    Converts non-numerical catorigal labels to numerical categorical labels.
    """
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    return y_train_enc

def creates_and_train_deep_learning_model():
    """Defining and training deep learning model."""
    np.random.seed(42)
    
    nb_epochs = 20
    
    # Deep learning model built in keras
    model = Sequential()
    
    model.add(Dense(1000,input_shape= (10000,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=nb_epochs, verbose=1)
    
    return model


def saving_model_to_file(model):
    """Model is converted to JSON file format and weights to HDF5."""
    # Serialize model to JSON.
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5.
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    
    # Defining variables
    path_to_train_dataset = 'train_data.csv'
    categorical_target_column = 'categorical_target_1'
    features_column = 'features'
    nb_classes = 43      # Chosen target variable, categorical_target_1 with 32 unique values or classes.
    
    # Reading data set
    dataset = pd.read_csv(path_to_train_dataset)
    
    # Dropping missing values
    dataset.dropna(subset=[categorical_target_column], inplace=True)
    
    # Pre-processing
    basic_preprocessing(dataset, features_column)
    
    # Feature Extraction: Term Frequency - Inverse Document Frequency (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', max_features= 10000,strip_accents='unicode', norm='l2')
    X_train = tfidf_vectorizer.fit_transform(dataset[features_column]).todense()
    
    # Preparing categorical target variable
    y_train_enc = prepare_targets(dataset[categorical_target_column])

    # Converts the 43 categories into one-hot encoding vectors in which 43 columns
    # are created and the values against the respective classes are given as 1. All other classes are given as 0.
    y_train = np_utils.to_categorical(y_train_enc, nb_classes)
    
    # Defining and training deep learning model.
    model = creates_and_train_deep_learning_model()
    
    # Model evaluation
    y_train_predclass = model.predict_classes(X_train,batch_size=BATCH_SIZE)
    
    print ("Deep Neural Network - Train Classification Report")
    print (classification_report(y_train_enc,y_train_predclass))
    
    # Saving model to file
    saving_model_to_file(model)
