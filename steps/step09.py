# 편의사항

import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None 
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # backward 메서드 간소화

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):        # output이 float64 같은 스칼라 타입인 경우 np.array(x)를 출력
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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

def square(x):
    f = Square()
    return f(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def exp(x):
    f = Exp()
    return f(x)

x = Variable(np.array(0.5))
# x = Variable(None): OK
# x = Variable(1.0): TypeError 발생
y = square(exp(square(x)))

y.backward()
print(x.grad)
