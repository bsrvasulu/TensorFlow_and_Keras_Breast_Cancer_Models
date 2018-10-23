# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:23:01 2018

@author: Sreenivasulu Bachu
"""


from dataKerasPreProcessor import *
from kerasModel import *

## MAIN APP CALL
if __name__ == '__main__':
    dataPreprocessor = dataKerasPreProcessor({'inputFileName': 'breast-cancer-wisconsin.csv'})
    X_train, Y_train, X_test, Y_test = dataPreprocessor.getData()
    print('------------------------------------------------------------------')
    print('Test started')
    kmodel = kerasModel(None)
    kmodel.retrieve_model(X_test, Y_test)   
    print('Test completed')
    print('------------------------------------------------------------------')    
