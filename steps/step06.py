# 수동 역전파

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # 미분값(grad)을 저장함

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input  # 입력변수를 보관(저장), backward(역전파)에서 사용하기 위해
        return output

    def forward(self, x):   # 순전파
        raise NotImplementedError()
    
    def backward(self, gy):   # 역전파
        raise NotImplementedError()

class Square(Function): 
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)

# We can see that our output value is similar with numerical diff value!