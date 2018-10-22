# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:27:02 2018

@author: Sreenivasulu Bachu
"""
import pandas as pd
import numpy as np
import sys

def loadData():
    try:
        df = pd.read_csv('breast-cancer-wisconsin.csv')
        df.drop(df[df['f'] == '?'].index, axis=0, inplace=True)
        df['f'] = df['f'].astype(np.int64)
        dfX = df.iloc[:, 1:10].copy()
        dfY = df.iloc[:, 10:11].copy()
        stdX = dfX.std(axis=0)
        X = dfX.values.T
        Y = dfY.values
        return X, Y.T, stdX;
        
    except:
        print("Unexpected error:", sys.exc_info()[0])

def randomData(X, Y):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)
    return shuffled_X, shuffled_Y

def getData():
    X,Y, stdX = loadData()
    #stdX = stdX.reshape((9,1))
    #print('stdX.shape = ', stdX.shape)
    #print('X.shape = ', X.shape)
    #print('Y.shape = ', Y.shape)
    #print('X[0] = ', X[0, 1])
    #print('Y[0] = ', Y[0, 0])
    
    # randomize data
    X, Y = randomData(X, Y)
    #print('X.shape = ', X.shape)
    #print('Y.shape = ', Y.shape)
    #print('X[0] = ', X[0, 1])
    #print('Y[0] = ', Y[0, 0])
    
    #conver Y to 0 and 1
    Y_ones = np.ones((1, Y.shape[1]))
    Y = Y_ones - (Y == 2)
   
    boundary = int(X.shape[1] * 0.7)
    X_train = X[:, 0: boundary]
    Y_train = Y[:, 0:boundary]
    X_train = X_train.astype(np.float)   
    X_train = X_train / 10.
    X_train = X_train.T
    Y_train = Y_train.astype(np.float)
    Y_train = Y_train.T
    print('X_train.shape = ', X_train.shape)
    print('Y_train.shape = ', Y_train.shape)
    
    X_test = X[:, boundary:]
    Y_test = Y[:, boundary:]
    X_test = X_test.astype(np.float)
    X_test = X_test / 10.
    X_test = X_test.T
    Y_test = Y_test.astype(np.float)
    Y_test = Y_test.T
    print('X_test.shape = ', X_test.shape)
    print('Y_test.shape = ', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

