import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def Full_Adder(X, Y, Z):
    SUM = XOR(XOR(X, Y), Z)
    CARRY = OR(AND(X, Y), AND(XOR(X, Y), Z))
    return CARRY, SUM

# 모든 결과 출력
print('Input (0, 0, 0) >> Carry, Sum', Full_Adder(0, 0, 0))
print('Input (0, 0, 1) >> Carry, Sum', Full_Adder(0, 0, 1))
print('Input (0, 1, 0) >> Carry, Sum', Full_Adder(0, 1, 0))
print('Input (0, 1, 1) >> Carry, Sum', Full_Adder(0, 1, 1))
print('Input (1, 0, 0) >> Carry, Sum', Full_Adder(1, 0, 0))
print('Input (1, 0, 1) >> Carry, Sum', Full_Adder(1, 0, 1))
print('Input (1, 1, 0) >> Carry, Sum', Full_Adder(1, 1, 0))
print('Input (1, 1, 1) >> Carry, Sum', Full_Adder(1, 1, 1))
