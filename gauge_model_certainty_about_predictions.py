#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:03:29 2020

@author: Ivomar Brito Soares
"""

import numpy as np


def predict_with_uncertainty(model, dataset, n_iter=10):
    """
    Calculates a NN model certainty/confidence, eg, when a NN tells 
    me an entry belongs to a certain category, I would like to know how certain it is.
    
    Args: 
         model: It's an object, which stores our neural network model's architecture. e.g. 
         how many dense layers will be used for training and the name of the function 
         used at output layer.
         
         dataset: The dataframe, which contains training data. e.g. Image or text data.
    
    """

    result = np.zeros((n_iter,) + dataset.shape)

    for iter in range(n_iter):
        result[iter] = model(dataset, 1)

    prediction  = result.mean(axis=0)
    uncertainty = result.var(axis=0)

    return prediction, uncertainty