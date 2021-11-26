# Import library
import  numpy as np
import json
from map_feature import map_feature
import matplotlib.pyplot as plt


# Compute_cost: calculate the cost of model of data set (the formula for calculating cost function is provided in “3. The formulas”).
def compute_cost(x, y, theta, lamda):
    m = len(x)
    Jl = 0
    Jr = 0

    for i in range(len(x)-1):
        h_theta = sigmoid(x[i], theta)
        Jl += -y[i] * np.log(h_theta) - (1-y[i]) * np.log(1 - h_theta)

    for j in range(len(theta)):
        Jr += theta[j]**2

    J = (1/m)*Jl + (lamda/(2*m))*Jr

    return J
    

def sigmoid(x, theta):
    return 1/(1 + np.exp(-(x @ theta.T)))


# Compute_gradient: calculate the gradient vector of the cost function (the formula for calculating the gradient vector is provided in “3. The formulas”).
def compute_gradient(x, y, theta, lamda):
    dJ = []
    m = len(x)

    for j in range(len(theta)):
        d_j = 0

        for i in range(len(x)):
            h_theta =  sigmoid(x[i],theta)
            d_j += (h_theta-y[i])*x[i][j]

        d_j *= 1/m
        if j != 0:
            d_j += (lamda/m) * theta[j]
        
        dJ.append(d_j)
        
    return dJ


# Gradient_descent: calculate the gradient descent.
def gradient_descent(x, y, theta, lamda, alpha): 
    dJ = compute_gradient(x, y, theta, lamda)

    for j in range(len(theta)):
        theta[j] -=  alpha*dJ[j]
    
    return theta


# Read the training configuration from file config.json
def read_training_configuration():
    #Load config
    with open('config.json',) as f:
        configs = json.load(f)
    return configs['Alpha'], configs['Lambda'], configs['NumIter']


# Read the training data from file training_data.txt
def read_training_data(): 
    datas = np.loadtxt('training_data.txt', delimiter = ',')
    
    # classify data into x, y
    x_raw = datas[: , 0:2]
    x1 = np.array(x_raw[:,0])
    x2 = np.array(x_raw[:,1])
    X0 = map_feature(x1, x2)
    x = np.c_[np.ones(len(X0),dtype='int64'), X0]

    y = datas[: , 2]

    theta = np.zeros(x.shape[1])

    return x_raw, x, y, theta


# Training data from file training_data.txt.
def model_fit(x, y, theta, alpha, lamda, numiter):
    for i in range(numiter):
        theta = gradient_descent(x, y, theta, lamda, alpha)
        cost = compute_cost(x, y, theta, lamda)
        
        print('Iter: {} - cost = {}'.format(i+1, cost))
    
    print('Train model...DONE!')

    return theta


# Save model to file model.json.
def save_model(theta): 
    model = {
        'theta': theta.tolist()
    }

    with open('model.json', 'w') as file:
        json.dump(model, file)
    
    print('Save model....DONE!')


# Predict: predict whether a set of microchips are eligible to be sold on market (pass an array of 1 element for prediction of 1 microchip).
# INPUT: [a, b] with a, b is feature
def predict(x, theta):
    h_theta = sigmoid(x, theta)
    if h_theta < 0.5:
        y = 0
    else:
        y = 1

    return y


# Calculate  accuracy  of  training  data  set
def calculate_accuracy(x, y, theta): 
    pred = []
    for i in range(len(x)):
        pred.append(predict(x[i], theta))
    correct = 0

    for i in range(len(y)):
        if pred[i] == y[i]:
            correct += 1
    accuracy =  (correct/len(y)) * 100

    print('Caculate accuracy...DONE!')
    return accuracy


# Make prediction and calculate accuracy of training data set, save result to file accuracy.json.
def save_predict_accuracy(x, y, theta, lamda): 
    accuracy = calculate_accuracy(x, y, theta)
    cost = compute_cost(x, y, theta, lamda)

    with open('accuracy.json', 'w') as file:
        result = {
            'Accuracy: ': accuracy,
            'Cost function:': cost
        }
        json.dump(result, file)

    print('Save accuracy...DONE!')
    return accuracy, cost


def ploting_decision_boundary(x_raw, y, theta):
    xs = np.linspace(min(x_raw[:,0]),max(y),num=10)
    ys = np.linspace(min(x_raw[:,0]),max(y),num=10)
    data0 = x_raw[y==0]
    data1 = x_raw[y==1]

    X_grid,Y_grid = np.meshgrid(xs,ys)
    Z_grid = np.zeros(shape=(len(ys),len(xs)))
    
    for i in range(len(xs)):
        for j in range(len(ys)):
            x1, x2 = [xs[i],ys[j]]

            # convert 2 feature to 28 feature
            X = map_feature(np.array([x1]),np.array([x2]))
            X = np.concatenate((np.array([[1]]),X),axis=1)

            value = predict(X, theta)
            Z_grid[j,i] = value
            
    plt.contourf(X_grid, Y_grid, Z_grid)
    plt.scatter(data0[:,0],data0[:,1],label='y=0')
    plt.scatter(data1[:,0],data1[:,1],label='y=1')
    plt.title('Logistic Regression\nPredict whether a factory microchip is eligible to be sold on market')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('Draw decision boundary...DONE!')


#-----------------------------------------------------------
# Main program:
if __name__ == '__main__':
    # - Read the training configuration from file config.json.
    alpha, lamda, numiter = read_training_configuration()
    
    # - Training data from file training_data.txt.
    x_raw, x, y, theta = read_training_data()
    theta = model_fit(x, y, theta, alpha, lamda, numiter)

    # - Save model to file model.json.
    save_model(theta)

    # - Make prediction and calculate accuracy of training data set, save result to file accuracy.json.
    accuracy, cost = save_predict_accuracy(x, y, theta, lamda)

    print('------------------------------')
    print('Cost function:', round(cost,3))
    print('Accuracy: {} %'. format(round(accuracy,3)))
    print('------------------------------')

    # Draw decision boundary
    ploting_decision_boundary(x_raw, y, theta)

