# 메모리 절약 모드
# 1. 역전파시, 불필요한 미분 결과를 보관하지 않고 즉시 삭제
# 2. '역전파가 필요 없는 경우용 모드'를 제공: 불필요한 계산 생략

# 추론(inference)(아마 testing을 말하는 것 같음)에선 순전파만 사용함 -> 역전파를 위해 순전파 값을 기억하고 있을 필요가 없음

import contextlib
import weakref
import unittest
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None 
        self.creator = None
        self.generation = 0  
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)   

        funcs = []
        seen_set = set()

        def add_func(f):  
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x: x.generation)
        
        add_func(self.creator)
        

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
        
            if not retain_grad:     # (기본값) 말단 변수 외에는 미분값을 유지하지 않음
                for y in f.outputs:
                    y().grad = None

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)   # 전처리
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)    # 후처리
    
def no_grad():
    return using_config('enable_backprop', False)

def as_array(x):     
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:     
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  
        ys = self.forward(*xs)  
        if not isinstance(ys, tuple):  
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]  

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   # (1) 세대 설정
            for output in outputs:
                output.set_creator(self)   # (2) 연결 설정
            self.inputs = inputs    # 순전파의 결과값을 기억하는 부분
            self.outputs = [weakref.ref(output) for output in outputs]     

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):  
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y, )

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function): 
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
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

def numerical_diff(f, x, eps = 1e-4):  
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SqaureTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self): 
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

#18.1
# x0 = Variable(np.array(1.0))
# x1 = Variable(np.array(1.0))
# t = add(x0, x1)
# y = add(x0, t)
# y.backward()

# print(y.grad, t.grad)
# print(x0.grad, x1.grad)

Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)