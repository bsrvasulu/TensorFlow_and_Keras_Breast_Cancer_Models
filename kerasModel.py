# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:59:03 2018

@author: Sreenivasulu Bachu
"""


#import numpy as np
#from keras import layers
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense

#import keras.backend as K
#import matplotlib.pyplot as plt
#import os
#import h5py
from keras.models import model_from_json


'''
# For a single-input model with 2 classes (binary classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)


# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)


import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''
class kerasModel:
    def __init__(self,params):
        #assign parameters
        self.params = params    
    
    def model_breastCancer(self, input_shape, network_shape):
        #define seqence model
        model = Sequential()
        model.add(Dense(network_shape[0], activation='relu', input_shape=(input_shape, )))   
        for i in range(1, len(network_shape)):
            model.add(Dense(network_shape[i], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    
    def fit_model(self, X_train, Y_train, X_test, Y_test, input_shape, network_shape, batch_size=32, epochs=200):
        # Create Model
        cancerModel = self.model_breastCancer(input_shape, network_shape)
        #cancerModel.compile(optimizer='adam',
        #          loss='mean_squared_error',
        #          metrics=['accuracy'])
        cancerModel.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        cancerModel.fit(X_train, Y_train, epochs, batch_size)
        
        # Evaluate
        preds = cancerModel.evaluate(x = X_test, y = Y_test)
        print()
        print("preds = " + str(preds))
        print("Loss = " + str(preds[0]))
        print("Accuracy = " + str(preds[1]))
        
        # save model
        # serialize model to JSON
        model_json = cancerModel.to_json()
        with open("./model_final/cancerModel.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        cancerModel.save_weights("./model_final/cancerModel.h5")
        print("Saved model to disk")
    
    def retrieve_model(self, X_test, Y_test):
        # load json and create model
        json_file = open('./model_final/cancerModel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./model_final/cancerModel.h5")
        print("Loaded model from disk")
        
        # compile
        #loaded_model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
        loaded_model.compile('rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Evaluate
        preds = loaded_model.evaluate(x = X_test, y = Y_test)
        print()
        print("preds = " + str(preds))
        print("Loss = " + str(preds[0]))
        print("Accuracy = " + str(preds[1]))