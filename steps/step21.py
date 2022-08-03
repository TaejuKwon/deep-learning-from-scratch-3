# 연산자 오버로드(2) 

import contextlib
import weakref
import unittest
import numpy as np

class Variable:
    __array_priority__ = 200    # Variable의 연산자 우선순위를 높여 좌항이 다른 타입이여도 우항(Variable type)을 기준으로 연산
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
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
        
            if not retain_grad: 
                for y in f.outputs:
                    y().grad = None

    @property   # x.shape() 대신 x.shape로 호출할 수 있음, shape 메소드를 인스턴스 변수처럼 사용할 수 있게 됨
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):      # len
        return len(self.data)
    
    def __repr__(self):     # print
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    # def __mul__(self, other):
    #     return mul(self, other) 

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)  
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)   
    
def no_grad():
    return using_config('enable_backprop', False)

def as_array(x):     
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:     
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]  
        ys = self.forward(*xs)  
        if not isinstance(ys, tuple):  
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]  

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   
            for output in outputs:
                output.set_creator(self)  
            self.inputs = inputs    
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
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

# Operation Overloading

Variable.__mul__ = mul
Variable.__add__ = add   
Variable.__rmul__ = mul
Variable.__radd__ = add     # 좌항이 Variable class가 아닌 경우에도 연산 가능케 함

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

x = Variable(np.array(2.0))

y = 3.0 * x + 1.0

print(y)
