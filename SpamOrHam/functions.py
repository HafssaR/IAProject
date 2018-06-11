# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:47:32 2018

@author: hafss
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)


def initialize_parameters(n_x, n_h, n_y):
    """retourne les poids initialisés de façon aléatoire
    """
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01 
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01 
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    



def linear_forward(A, W, b):
    """
    la propagation des poids de l'entrée à la sortie avec l'entrée A, le poids, et le biais b
    """
    Z = np.dot(W,A)+b   
    cache = (A, W, b)   
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """selon si la fonction d'activation est la fonction sigmoid ou relu, retourne 
    A la sortie d'un neurone du hidden layer"""
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
       
    cache = (linear_cache, activation_cache)

    return A, cache




def compute_cost(AL, Y):
    """
    calcul cost function cad l'estimation de l'erreur
    """
    
    m = Y.shape[1]

    cost = - np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m
    
    cost = np.squeeze(cost)
    
    return cost



def linear_backward(dZ, cache):
    """
    la propagation inverse pour le calcul des gradients et ainsi pouvoir ajuster
    les poids et les biais
    """
    A_p, W, b = cache
    m = A_p.shape[1]
    dW = np.dot(dZ,A_p.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_p = np.dot(W.T,dZ)
    
    return dA_p, dW, db




def linear_activation_backward(dA, cache, activation):
    """
    Implemente backward propagation pour LINEAR->ACTIVATION layer.  
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
  
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        
    elif activation == "sigmoid":
      
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
       
    
    return dA_prev, dW, db



def update_parameters(parameters, grads, learning_rate):
    """
    ajustement des parametres
    """
    
    L = len(parameters) // 2 # number of layers in the neural network
   
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W"+ str(l)]-learning_rate*grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b"+ str(l)]-learning_rate*grads["db" + str(l)]
    
    return parameters




