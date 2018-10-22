# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:33:11 2018

@author: Sreenivasulu Bachu
"""
from tfDataPreProcessor import getData
from tfModel import restore_model

      
## MAIN APP CALL
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getData()   
    restore_model(X_test, Y_test, [9,8,8,4,4,1])    
