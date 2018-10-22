# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:55:20 2018

@author: Sreenivasulu Bachu
"""


from dataKerasPreProcessor import getData
from kerasModel import fit_model

## MAIN APP CALL
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getData()
    print('------------------------------------------------------------------')
    print('Train started')
    fit_model(X_train, Y_train, X_test, Y_test, input_shape = 9, network_shape = [8,8,4,4], batch_size = 32, epochs = 2000)   
    print('Train completed')
    print('------------------------------------------------------------------')    
