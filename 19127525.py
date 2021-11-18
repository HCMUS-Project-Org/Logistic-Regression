# Import library
import  numpy as np
import json
from math import log, exp
from map_feature import map_feature

# Compute_cost: calculate the cost of model of data set (the formula for calculating cost function is provided in “3. The formulas”).
def compute_cost(x, y, theta, lamda): #DONE_not test yet!
    m = len(x)
    Jl = 0
    Jr = 0
    # print('-------------------------')
    # print(x)
    # print('len:',len(x),len(x[0]))
    # print(theta)
    # print('len:', len(theta))

    for i in range(len(x)-1):
        # print(i)
        # print('x0:',x[0])
        # print('theta0:',theta)
        # h_theta =  1 / (1 + exp(-(x[i] @ theta.T)))
        h_theta = sigmoid(x[i], theta)
        Jl += -y[i] * np.log(h_theta) - (1-y[i]) * np.log(1 - h_theta)

    for j in range(len(theta)):
        Jr += theta[j]**2

    J = (1/m)*Jl + (lamda/(2*m))*Jr

    return J
    

def sigmoid(x, theta):
    return 1/(1 + np.exp(-(x @ theta.T)))

# Compute_gradient: calculate the gradient vector of the cost function (the formula for calculating the gradient vector is provided in “3. The formulas”).
def compute_gradient(x, y, theta, lamda): #DONE_not test yet!
    dJ = []
    m = len(x)

    for j in range(len(theta)):
        d_j = 0

        for i in range(len(x)):
            # h_theta =  1 / (1 + exp(-(x[i] @ theta.T)))
            h_theta =  sigmoid(x[i],theta)
            d_j += (h_theta-y[i])*x[i][j]

        d_j *= 1/m
        if j != 0:
            d_j += (lamda/m) * theta[j]
        
        dJ.append(d_j)
        
    return dJ


# Gradient_descent: calculate the gradient descent.
def gradient_descent(x, y, theta, lamda, alpha): #DONE_not test yet
    dJ = compute_gradient(x, y, theta, lamda)

    for j in range(len(theta)):
        theta[j] -=  alpha*dJ[j]
    
    return theta

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
    x_raw = datas[: , 0:2]
    x1 = np.array(x_raw[:,0])
    x2 = np.array(x_raw[:,1])
    X0 = map_feature(x1, x2)

    x = np.c_[np.ones(len(X0),dtype='int64'), X0]

    # x = np.c_[np.ones(len(x_raw),dtype='int64'), x_raw]

    y = datas[: , 2]
    theta = np.zeros(x.shape[1])

    return x, y, theta

# Training data from file training_data.txt.
def model_fit(x, y, theta, alpha, lamda, numiter):
    for i in range(numiter):
        theta = gradient_descent(x, y, theta, lamda, alpha)
        cost = compute_cost(x, y, theta, lamda)
        
        print('Iter: {} - cost function = {}'.format(i+1, cost))

    return theta

# Save model to file model.json.
def save_model(theta): #DONE
    model = {
        'theta': theta.tolist()
    }

    with open('model.json', 'w') as file:
        json.dump(model, file)
    print('Save....DONE')

# Predict: predict whether a set of microchips are eligible to be sold on market (pass an array of 1 element for prediction of 1 microchip).
# INPUT: [a, b] with a, b is feature
def predict(x, theta): #DONE
    print('-------------------------')
    print(x)
    print('len:',len(x))
    print(theta)
    print('len:', len(theta))
    h_theta = sigmoid(x, theta)
    if h_theta < 0.5:
        y = 0
    else:
        y = 1

    return y

# Calculate  accuracy  of  training  data  set
def calculate_accuracy(x, y, theta): #DONE
    pred = []
    for i in range(len(x)):
        pred.append(predict(x[i], theta))
    correct = 0

    for i in range(len(y)):
        if pred[i] == y[i]:
            correct += 1
    accuracy =  (correct/len(y)) * 100
    return accuracy

# Make prediction and calculate accuracy of training data set, save result to file accuracy.json.
# x_predict = [0.2, 0.05] INPUT 1 microchip feature
def save_predict_accuracy(x_predict, x, y, theta, lamda): #DONE_not test yet
    x_predict.insert(0,1)
    x_predict = np.array(x_predict)

    accuracy = calculate_accuracy(x, y, theta)
    cost = compute_cost(x, y, theta, lamda)
    # pred =  predict(x_predict, theta)
    pred = 0

    with open('accuracy.json', 'w') as file:
        result = {
            'Feature 1: ': x_predict[0],
            'Feature 2: ': x_predict[1],
            'Eligible: ': pred,
            'Accuracy: ': accuracy,
            'Cost function:': cost
        }
        json.dump(result, file)

    # return accuracy, cost
    return accuracy, cost, pred

#-----------------------------------------------------------
# Main program:
if __name__ == '__main__':
    # - Read the training configuration from file config.json.
    alpha, lamda, numiter = read_training_configuration()
    
    # - Training data from file training_data.txt.
    x, y, theta = read_training_data()
    theta = model_fit(x, y, theta, alpha, lamda, numiter)

    # print(theta)
    # - Save model to file model.json.
    save_model(theta)

    # - Make prediction and calculate accuracy of training data set, save result to file accuracy.json.
    x_predict = [0.0513, 0.6996]
    accuracy, cost, pred = save_predict_accuracy(x_predict, x, y, theta, lamda)


    print('Feature 1:', x_predict[0])
    print('Feature 2:', x_predict[1])
    print('Predict:', pred)
    print('----------------------')
    print('Cost function:', cost)
    print('Accuracy:', accuracy)