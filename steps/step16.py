# 가변 길이 인수(역전파)
# 순전파: 입력 2, 출력 1 -> 역전파: 입력 1, 출력 2

# 가변 길이 인수(개선)

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
        self.generation = 0     # 역전파시 함수 처리의 우선순위
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)   

        funcs = []
        seen_set = set()

        def add_func(f):    # 함수들을 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x: x.generation)
        
        add_func(self.creator)
        

        while funcs:
            f = funcs.pop()
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                # x.grad = gx     # 미분값을 단순히 덮어씌우기 때문에, 같은 변수를 사용할 경우 오류 발생
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

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

        self.generation = max([x.generation for x in inputs])   # inputs 중 가장 큰 geneartion을 따라감
        for output in outputs:
            output.set_creator(self)   
        self.inputs = inputs
        self.outputs = outputs

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

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)