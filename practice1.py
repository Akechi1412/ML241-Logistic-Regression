import numpy as np
import matplotlib.pyplot as plt

# Initial Values
x = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.306, 1.045], dtype='float64')
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype='float64')
theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = 0.9
epochs = 5000
epsilon = 1e-8

# Sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Cost Function
def cost(x, y, theta0, theta1):
    m = x.shape[0]
    y_hat = sigmoid(theta0 + theta1*x)
    return -1/m * np.sum(y*np.log(y_hat + 1e-15) + (1 - y)*np.log(1 - y_hat + 1e-15))

# Gradient Descent
def gradient_descent(x, y, theta0, theta1):
    m = x.shape[0]
    y_hat = sigmoid(theta0 + theta1*x)
    theta0 = theta0 - learning_rate * 1/m * np.sum((y_hat - y)) 
    theta1 = theta1 - learning_rate * 1/m * np.sum((y_hat - y) * x) 
    return theta0, theta1

if __name__ == '__main__':
    delta_j = np.Infinity
    for i in range(epochs):
        if abs(delta_j) > epsilon:
            j = cost(x, y, theta0, theta1)
            theta0, theta1 = gradient_descent(x, y, theta0, theta1)
            j_new = cost(x, y, theta0, theta1)
            delta_j = j_new - j
            j = j_new
            print(f'iterators: {i+1}, cost: {j}')
        else:
            break

    print(f'\ntheta0: {theta0}, theta1: {theta1}')

    # Display the plot
    fig, ax = plt.subplots()
    ax.set(xlabel='Grant size(mm)', ylabel='Spiders(present: 1, absent: 0)')
    ax.axis([0, 2, -1, 2])
    ax.scatter(x, y, s=100, facecolor='navy')
    x = np.arange(0, 500, 1)
    y_hat = sigmoid(theta0 + theta1*x)
    ax.plot(x, y_hat)
    plt.show()