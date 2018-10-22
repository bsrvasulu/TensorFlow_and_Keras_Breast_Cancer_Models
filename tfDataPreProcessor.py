# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:19:20 2018

@author: Sreenivasulu Bachu
"""
import pandas as pd
import numpy as np
import sys

class tfDataPreProcessor:
    def __init__(self,params):
        #assign parameters
        self.params = params
        self.inputFileName = params['inputFileName']
        
    def loadData(self):
        try:
            #df = pd.read_csv('breast-cancer-wisconsin.csv')
            df = pd.read_csv(self.inputFileName)
            #df = pd.read_csv('breast-cancer-wisconsin.csv', header=None)
            df = df.ix[df['f'] != '?']
            dfX = df.iloc[:, 1:10].copy()
            dfY = df.iloc[:, 10:11].copy()
            X = dfX.values.T
            Y = dfY.values
            return X, Y.T;
            
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
    def randomData(self, X, Y):
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)
        return shuffled_X, shuffled_Y
    
    def getData(self):
        X,Y = self.loadData()
        #print('X.shape = ', X.shape)
        #print('Y.shape = ', Y.shape)
        #print('X[0] = ', X[0, 1])
        #print('Y[0] = ', Y[0, 0])
        
        # randomize data
        X, Y = self.randomData(X, Y)
        #print('X.shape = ', X.shape)
        #print('Y.shape = ', Y.shape)
        #print('X[0] = ', X[0, 1])
        #print('Y[0] = ', Y[0, 0])
        
        #conver Y to 0 and 1
        Y_ones = np.ones((1, Y.shape[1]))
        Y = Y_ones - (Y == 2)
        #print("Y:", Y)
        
        boundary = int(X.shape[1] * 0.7)
        X_train = X[:, 0: boundary]
        Y_train = Y[:, 0:boundary]
        X_train = X_train.astype(np.float)
        X_train = X_train / 10.
        Y_train = Y_train.astype(np.float)
        print('X_train.shape = ', X_train.shape)
        print('Y_train.shape = ', Y_train.shape)
        
        X_test = X[:, boundary:]
        Y_test = Y[:, boundary:]
        X_test = X_test.astype(np.float)
        X_test = X_test / 10.
        Y_test = Y_test.astype(np.float)
        print('X_test.shape = ', X_test.shape)
        print('Y_test.shape = ', Y_test.shape)
        
        return X_train, Y_train, X_test, Y_test
    
'''       
## MAIN APP CALL
if __name__ == '__main__':
    #restore_model()    
    X,Y = loadData()
    print('X.shape = ', X.shape)
    print('Y.shape = ', Y.shape)
    print('X[0] = ', X[0, 1])
    print('Y[0] = ', Y[0, 0])
    
    # randomize data
    X, Y = randomData(X, Y)
    print('X.shape = ', X.shape)
    print('Y.shape = ', Y.shape)
    print('X[0] = ', X[0, 1])
    print('Y[0] = ', Y[0, 0])
    
    #conver Y to 0 and 1
    Y_ones = np.ones((1, Y.shape[1]))
    Y = Y_ones - (Y == 2)
    #print("Y:", Y)
    
    boundary = int(X.shape[1] * 0.7)
    X_train = X[:, 0: boundary]
    Y_train = Y[:, 0:boundary]
    X_train = X_train.astype(np.float)
    X_train = X_train / 10.
    Y_train = Y_train.astype(np.float)
    print('X_train.shape = ', X_train.shape)
    print('Y_train.shape = ', Y_train.shape)
    
    X_test = X[:, boundary:]
    Y_test = Y[:, boundary:]
    X_test = X_test.astype(np.float)
    X_test = X_test / 10.
    Y_test = Y_test.astype(np.float)
    print('X_test.shape = ', X_test.shape)
    print('Y_test.shape = ', Y_test.shape)
    
    #data = np.random.random((1000, 100))
    #labels = np.random.randint(2, size=(1000, 1))
    #print(data.shape)
    parameters = model(X_train, Y_train, X_test, Y_test, [9,8,8,4,4,1])    
    print('parameters: ', parameters)
    
    #if save_weights == True:
    #    model_parameters_json = json.dumps(parameters2)
    #    with open("model/MODEL.json", "w") as json_file:
    #        json_file.write(model_parameters_json)
    
'''