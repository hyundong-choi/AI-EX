import numpy as np
import matplotlib.pyplot as plt

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    # 가중치와 임계값의 중요성....
    # 적절한 가중치를 찾기 위한 방법이 학습.
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        # print("AND = 0")
        return 0
    elif tmp > theta:
        # print("AND = 1")
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        # print("NAND = 0")
        return 0
    else:
        # print("NAND = 1")
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        # print("OR = 0")
        return 0
    else:
        # print("OR = 1")
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    print("XOR = ", y)
    return y