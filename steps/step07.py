# 역전파 자동화
# Define-by-Run = 동적 계산 그래프

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None 
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()    # 하나 앞 변수의 backward 함수를 호출 (재귀)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input 
        self.output = output
        return output


    def forward(self, x):  
        raise NotImplementedError()
    
    def backward(self, gy):
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

# 계산 그래프의 노드들을 거꾸로 거슬러 올라간다
# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a



# 1. 함수를 가져온다
# 2. 함수의 입력을 가져온다
# 3. 함수의 backward 메소드를 호출한다

y.grad = np.array(1.0)

# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)

# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)

y.backward()

print(x.grad)