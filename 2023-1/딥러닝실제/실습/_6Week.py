import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrint():
    x = np.array([-1.0, 1.0, 2.0])
    sigmoid(x)
    print(sigmoid(x))

def sigmoidPrint_2():
    t = np.array([1.0, 2.0, 3.0])
    1.0 + t
    print(1.0 + t)
    1.0 / t
    print(1.0 / t)

def _6WeekDrawGraph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def matrixMul():
    A = np.array([[1, 2], [3, 4]])
    print("A shape : ", A.shape)
    B = np.array([[5, 6], [7, 8]])
    print("B shape : ", B.shape)

    print(np.dot(A, B))

def _3floorNN():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    print("W1 shape : ", W1.shape)
    print("X shape : ", X.shape)
    print("B1 shape : ", B1.shape)

    A1 = np.dot(X, W1) + B1

    Z1 = sigmoid(A1)

    print("A1 : ", A1)
    print("Z1 : ", Z1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    print("Z1 shape : ", Z1.shape)
    print("W2 shape : ", W2.shape)
    print("B2 shape : ", B2.shape)

    A2 = np.dot(Z1, W1) + B2
    Z2 = sigmoid(A2)

    print("A2 : ", A2)
    print("Z2 : ", Z2)

def softMax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softMax():
    a = np.array([0.3, 2.9, 4.0])
    exp_a = np.exp(a)
    print(exp_a)

    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)

    y = exp_a / sum_exp_a
    print(y)


