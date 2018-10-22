# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:11 2018

@author: Sreenivasulu Bachu
"""


from tfDataPreProcessor import *
from tfModel import *

if __name__ == '__main__':
    model = tfModel(None)
    dataPreprocessor = tfDataPreProcessor({'inputFileName': 'breast-cancer-wisconsin.csv'})    
    X_train, Y_train, X_test, Y_test = dataPreprocessor.getData()
    
    # network_shape - first one should match with input parameters
    parameters, costs = model.model(X_train, Y_train, X_test, Y_test, network_shape = [9,8,8,4,4,1])    
    #print('parameters: ', parameters)
    
