# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:11 2018

@author: Sreenivasulu Bachu
"""


from tfDataPreProcessor import getData
from tfModel import model

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getData()
    
    # network_shape - first one should match with input parameters
    parameters, costs = model(X_train, Y_train, X_test, Y_test, network_shape = [9,8,8,4,4,1])    
    print('parameters: ', parameters)
    
