# 메모리 관리와 순환 참조

# 1. reference의 수를 세는 '참조 카운트'
# 2. 세대를 기준으로 쓸모 없어진 객체를 회수하는 'GC(Garbage Collection)'

# 17.1 참조 카운트 방식의 메모리 관리

# 17.2 순환 참조시 a = b = c = None 을 실행했을 때, 모든 참조 카운트가 0이 되지 않음
# -> GC

# 순환 참조로 인해 생기는 잉여 메모리 해제를 GC에 미루다 보면 프로그램의 전체 메모리 사용량이 증가함
# 이는 신경망, 머신러닝에서 매우 치명적인 사안

# weakref 구조: "약한 참조": 다른 객체를 참조하되 참조 카운트는 증가시키지 않음

# 가변 길이 인수(역전파)
# 순전파: 입력 2, 출력 1 -> 역전파: 입력 1, 출력 2

# 가변 길이 인수(개선)

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

    def backward(self):
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

        self.generation = max([x.generation for x in inputs]) 
        for output in outputs:
            output.set_creator(self)   
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]      # self.outputs가 대상을 약한 참조로 가리키게 변경
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

for i in range(10):
    x = Variable(np.random.randn(10000))    # 거대한 데이터
    y = square(square(square(x)))       # 복잡한 계산

# 순환 참조 문제를 야기하는 상황
# for문이 다음 번째로 넘어 갈 때, x와 y가 덮어 써짐. 이전의 계산 그래프를 더 이상 참조하지 않음
# 참조 카운트 0 -> 이전 반복문에서의 계산 메모리 바로 삭제