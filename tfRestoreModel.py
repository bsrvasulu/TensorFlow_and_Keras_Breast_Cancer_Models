# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:33:11 2018

@author: Sreenivasulu Bachu
"""
from tfDataPreProcessor import *
from tfModel import *

      
## MAIN APP CALL
if __name__ == '__main__':
    model = tfModel(None)
    dataPreprocessor = tfDataPreProcessor({'inputFileName': 'breast-cancer-wisconsin.csv'})    
    X_train, Y_train, X_test, Y_test = dataPreprocessor.getData()   
    model.restore_model(X_test, Y_test, [9,8,8,4,4,1])    
