import  numpy as np
import json
from math import log
from map_feature import map_feature

# - compute_cost: calculate the cost of model of data set (the formula for calculating cost function is provided in “3. The formulas”).
def compute_cost(x, y, theta, lamda):
    m = len(x)
    Jl = 0
    Jr = 0

    for i in range(len(x)):
        h_theta = x[i] @ theta.T
        Jl += -y[i] * log(h_theta) - (1-y[i]) * log(1 - h_theta)

    for j in range(len(theta)):
        Jr += theta[j]**2

    J = (1/m)*Jl + (lamda/(2*m))*Jr

    return J
    
# - compute_gradient: calculate the gradient vector of the cost function (the formula for calculating the gradient vector is provided in “3. The formulas”).
def compute_gradient():
    pass

# - gradient_descent: calculate the gradient descent.
def gradient_descent():
    pass

# - predict: predict whether a set of microchips are eligible to be sold on market (pass an array of 1 element for prediction of 1 microchip).
def predict():
    pass

# Read the training configuration from file config.json
def read_training_configuration(): #DONE
    #Load config
    with open('config.json',) as f:
        configs = json.load(f)
    return configs['Alpha'], configs['Lambda'], configs['NumIter']

# Read the training data from file training_data.txt
def read_training_data(): #DONE
    datas = np.loadtxt('training_data.txt', delimiter = ',')
    
    # classify data into x, y
    x = datas[: , 0:2]
    x = np.c_[np.ones(len(x),dtype='int64'), x]

    y = datas[: , 2]
    theta = np.zeros(x.shape[1])
    return x, y, theta

# Training data from file training_data.txt.
def training_data():
    pass

# Save model to file model.json.
def save_model():
    pass

# Make prediction of training data set, save result to file accuracy.json.
def prediction():
    pass

# Calculate accuracy of training data set, save result to file accuracy.json.
def calculate_accuracy():
    pass

#-----------------------------------------------------------
# Main program:
if __name__ == '__main__':
    pass
# - Read the training configuration from file config.json.

# - Training data from file training_data.txt.

# - Save model to file model.json.

# - Make prediction and calculate accuracy of training data set, save result to file accuracy.json.