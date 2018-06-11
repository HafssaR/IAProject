# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:53:15 2018

@author: hafss
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import os
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)


"""LOADING X AND Y """

Y=np.zeros((1,4001))
X= np.zeros((1899,4001))
spam=0
ham=0
k=0
for filename in os.listdir("D:/IAProjet/ProjetIA/TrainSet"):
    
    if filename.endswith("spam.txt"): 
        k+=1
        spam+=1
        print(os.path.join("D:/IAProjet/ProjetIA/TrainSet", filename))
        file= open(os.path.join("D:/IAProjet/ProjetIA/TrainSet", filename), "r")
        data=file.read().replace('\n', '')
        file.close()
      
        for i in indices_mots(data):  
            X[i][k]=1
        Y[0][k]=1
    elif filename.endswith("ham.txt"): 
        ham+=1
        print(os.path.join("D:/IAProjet/ProjetIA/TrainSet", filename))
        k+=1
        file= open(os.path.join("D:/IAProjet/ProjetIA/TrainSet", filename), "r")
        data=file.read().replace('\n', '')
        file.close()
        for j in indices_mots(data):
            X[j][k]=1
        Y[0][k]=0      

train_x=X
train_y=Y 

Yt =np.zeros((1,1678))
Xt= np.zeros((1899,1678))
l=0
spam=0
ham=0
for filename in os.listdir("D:/IAProjet/ProjetIA/TestSet"):
    if filename.endswith("spam.txt"): 
        spam+=1
        l+=1
        print(os.path.join("D:/IAProjet/ProjetIA/testSet", filename))
        file= open(os.path.join("D:/IAProjet/ProjetIA/TestSet", filename), "r")
        data=file.read().replace('\n', '')
        file.close()
        
        for i in indices_mots(data):
            Xt[i][l]=1
        Yt[0][l]=1
    elif filename.endswith("ham.txt"): 
        ham+=1
        print(os.path.join("D:/IAProjet/ProjetIA/testSet", filename))
        l+=1
        file= open(os.path.join("D:/IAProjet/ProjetIA/TestSet", filename), "r")
        data=file.read().replace('\n', '')
        file.close()
        for j in indices_mots(data):
            Xt[j][l]=1
        Yt[0][l]=0    
 
        
test_x = Xt
test_y = Yt
print(test_x.shape)
print(test_y.shape)


n_x = 1899
n_y = 1
n_h = 7

layers_dims = (n_x, n_h, n_y)
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                             
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
   
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. 
        
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 =linear_activation_forward(A1, W2, b2, activation='sigmoid')
        
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    
        # Backward propagation. 
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    print(Y)
    print(A2)
    return parameters

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 3500, print_cost=True)

def test(X, Y, parameters):
    ypred = np.zeros(Y.shape)
    s=0
    VP=0
    FP=0
    VN=0
    FN=0
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]    
    
    A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
    A2, cache2 =linear_activation_forward(A1, W2, b2, activation='sigmoid')
    
    i = A2.shape[1]

    for j in range(i-1):
        if A2[0][j]>0.5:
            ypred[0][j]=1
    print(ypred)
    print(Y)
    for q in range(i-1):
        if (ypred[0][q]== Y[0][q]):
            s+=1
            if (Y[0][q] ==1):
                VP+=1
            else:
                VN+=1
        else:
            if (Y[0][q] ==1):
                FP+=1
            else: 
                FN+=1
    precision = VP/(VP+FP)
    rappel = VP/(VP+FN)
    F = (2*precision*rappel)/(rappel+precision)
    print("Precision : {}" .format(precision))
    print("Rappel : {}" .format(rappel))   
    print("F score: {}" .format(F))            
    print(s/i)

test(test_x,test_y, parameters)
    
    
def predict(X, parameters):
    ypred = np.zeros((1,1))
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
     
    A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
    A2, cache2 =linear_activation_forward(A1, W2, b2, activation='sigmoid')
    
    i = A2.shape[1]

    for j in range(i):
        if A2[0][j]>0.5:
            ypred[0][j]=1
    return ypred

def test_unitaire(path, parameters):
    file = open(path, "r")
    euf=file.read().replace('\n', '')
    file.close()      
    ind = indices_mots(euf)
    Xtry = np.zeros((1899,1))
    for h in ind:
        Xtry[h]=1
    return predict( Xtry,parameters )



print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/spam6.txt", parameters))
print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/spam8.txt", parameters))
print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/spam9.txt", parameters))
print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/spam10.txt", parameters))
print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/0046.2003-12-20.GP.spam.txt", parameters))
print(test_unitaire("D:/IAProjet/ProjetIA/spam_test/0290.2000-01-23.kaminski.ham.txt", parameters))




 








    
    
    
    
    
    
