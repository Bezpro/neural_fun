import numpy as np 
import nn as n

def sigmoid(x):
    return 1 / (1 + np.power(np.exp(1),-x))

def hyperbolic_tangent(x):
    return (np.power(np.exp(1),2*x)-1)/(np.power(np.exp(1),2*x)+1)

def sigmoid_derivative(x):
        return sigmoid(x)*(1-sigmoid(x)) 

