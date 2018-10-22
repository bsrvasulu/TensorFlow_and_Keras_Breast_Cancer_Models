# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:23:01 2018

@author: Sreenivasulu Bachu
"""


from dataKerasPreProcessor import getData
from kerasModel import retrieve_model

## MAIN APP CALL
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getData()
    print('------------------------------------------------------------------')
    print('Test started')
    retrieve_model(X_test, Y_test)   
    print('Test completed')
    print('------------------------------------------------------------------')    
