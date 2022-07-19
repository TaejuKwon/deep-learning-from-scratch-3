# 미세한 차이를 이용한 미분: 수치 미분(numerical diff) / example: h = 1e-4 (컴퓨터는 극한을 취급할 수 없음)
# 중앙차분이 오차가 더 적음 (f(x+h) - f(x-h) / 2h)

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function): 
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def numerical_diff(f, x, eps = 1e-4):       # epsilon = h
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)

def f(x):       # 합성함수의 수치미분
    A = Square()
    B = Exp()
    C = Square()
    return (C(B(A(x))))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
