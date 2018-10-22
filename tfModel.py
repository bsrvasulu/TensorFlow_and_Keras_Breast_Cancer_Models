# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:49:29 2018

@author: Sreenivasulu Bachu
"""


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math

class tfModel:
    def __init__(self,params):
        #assign parameters
        self.params = params
        
    def random_minibatches(self, X, Y, minimatch_size = 32):
        m = X.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)
        
        num_complete_mnibatches = math.floor(m/minimatch_size)
        for k in range (0,  num_complete_mnibatches):
            mini_batch_X = shuffled_X[:, k * minimatch_size : k * minimatch_size + minimatch_size]
            mini_batch_Y = shuffled_Y[:, k * minimatch_size : k * minimatch_size + minimatch_size]
        
            mini_batch = (mini_batch_X, mini_batch_Y)        
            mini_batches.append(mini_batch)
        
        # add final one with remaining data (less than mini batch size)
        if m % minimatch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_mnibatches * minimatch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_mnibatches * minimatch_size : m]
        
            mini_batch = (mini_batch_X, mini_batch_Y)        
            mini_batches.append(mini_batch) 
            
        return mini_batches
    
    def create_place_holders(self, n_x, n_y):
        X = tf.placeholder(tf.float32, shape = (n_x, None), name='X')
        Y = tf.placeholder(tf.float32, shape = (n_y, None), name='Y')
        return X, Y
    
    
    def initiliaze_weights_bias(self, network_shape):
        parameters = {}
        for i in range(1, len(network_shape)):
            parameters['W' + str(i)] = tf.get_variable('W' + str(i), [network_shape[i], network_shape[i-1]], initializer = tf.contrib.layers.xavier_initializer())
            parameters['b' + str(i)] = tf.get_variable('b' + str(i), [network_shape[i], 1], initializer = tf.zeros_initializer())
    
        return parameters
    
    def initiliaze_weights_bias2(self, network_shape):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [8, 9]
                            b1 : [8, 1]
                            W2 : [4, 8]
                            b2 : [4, 1]
                            W3 : [1, 4]
                            b3 : [1, 1]
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        
        tf.set_random_seed(1)                              # so that your "random" numbers match ours
            
        ### START CODE HERE ### (approx. 6 lines of code)
        W1 = tf.get_variable("W1", [8, 9], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [8,1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [4,8], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [4,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [1,4], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
        ### END CODE HERE ###
    
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        return parameters
    
    def farward_propagation(self, X, parameters):    
        networkLayers = int(len(parameters) / 2)
        # calculate first value
        ZT = tf.add(tf.matmul(parameters['W1'], X), parameters['b1'])
        AT = tf.nn.relu(ZT)
        for i in range(2, networkLayers):
            ZT = tf.add(tf.matmul(parameters['W' + str(i)], AT), parameters['b' + str(i)])
            AT = tf.nn.relu(ZT)
        ZT = tf.add(tf.matmul(parameters['W' + str(networkLayers)], AT), parameters['b' + str(networkLayers)])
        
        return ZT
    
    def compute_cost2(self, Z, Y):
        logits = tf.transpose(Z)
        labels = tf.transpose(Y)    
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))    
        return cost
    
    def compute_cost(self, finalZ, Y):
        logits = tf.transpose(finalZ)
        labels = tf.transpose(Y)
        
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
        return cost
    
    def model(self, X_train, Y_train, X_test, Y_test, network_shape, learning_rate = 0.0001, num_epochs = 200, minibatch_size = 32, print_cost = True, save_weights = True):
        ops.reset_default_graph()
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        
        #Create place holders to hold features and output (X and Y)
        X, Y = self.create_place_holders(n_x, n_y)
        
        # Initialize weights and biases
        parameters = self.initiliaze_weights_bias(network_shape)
        
        #farward propagation 
        Z_final = self.farward_propagation(X, parameters)
        
        #compute cost
        cost = self.compute_cost(Z_final, Y)
        
        #optimize
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        
        # Create a Saver object
        saver = tf.train.Saver()
    
        #initialize all variable
        init = tf.global_variables_initializer()
        
        #run the tf session to do the process
        with tf.Session() as sess:
            #Run the initializer
            sess.run(init)
            
            #Do the training loop
            for epoach in range(num_epochs):
                epoach_cost = 0
                #Get random mini batches
                minibatches  = self.random_minibatches(X_train, Y_train, minibatch_size) 
                
                #Get num of mini batches
                num_minibatches = len(minibatches)
                
                for minibatch in minibatches:
                    #Select minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    #Run session to execute optimizer
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                    
                    epoach_cost += minibatch_cost/num_minibatches
                    
                #Print cost
                if print_cost == True and epoach%20 == 0:
                    print('Cost after epoach %i: %f' %(epoach, epoach_cost))
                if print_cost == True and epoach%5 == 0:
                    costs.append(epoach_cost)
                    
            # Save the final model
            saver.save(sess, './model_final/model')
            
            #Get and Save parameters 
            parameters = sess.run(parameters)
            print('Parameters have been trained!')
            
            #plot cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel("Iteration (per 5s)")
            plt.title('Learning rate: %f' %(learning_rate))
            plt.show()
            
            # Calculate Correct predictions
            prediction = tf.equal(tf.greater(Z_final, tf.constant(0.5)), tf.greater(Y, tf.constant(0.5)))
            accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            
        #if save_weights == True:
        #    model_parameters_json = json.dumps(parameters2)
        #    with open("model/MODEL.json", "w") as json_file:
        #        json_file.write(model_parameters_json)
    
        return  parameters, costs               
                    
            
    def restore_model(self, X_test, Y_test, network_shape):
        tf.reset_default_graph()  
        parameters = {}
        imported_meta = tf.train.import_meta_graph("./model_final/model.meta")  
        with tf.Session() as sess:
            imported_meta.restore(sess, tf.train.latest_checkpoint('./model_final/'))                      
                    
            graph = tf.get_default_graph()
            W1 = graph.get_tensor_by_name("W1:0")
            W2 = graph.get_tensor_by_name("W2:0")        
            print ("W1:", sess.run(W1))
            print ("W2:", sess.run(W2))    
            
            for i in range(1, len(network_shape)):
                parameters['W' + str(i)] = sess.run(graph.get_tensor_by_name("W"+ str(i)+":0"))
                parameters['b' + str(i)] = sess.run(graph.get_tensor_by_name("b"+ str(i)+":0"))
    
        
        (n_x, m) = X_test.shape
        n_y = Y_test.shape[0]
        
        # Create input place holders
        X, Y = self.create_place_holders(n_x, n_y)
        
        #Forward propagation
        Z_final = self.farward_propagation(X, parameters)
    
        with tf.Session() as sess:   
            # Calculate Correct predictions
            prediction = tf.equal(tf.greater(Z_final, tf.constant(0.5)), tf.greater(Y, tf.constant(0.5)))
            accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))    
        
        