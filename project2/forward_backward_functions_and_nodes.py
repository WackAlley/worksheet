from __future__ import annotations
from itertools import combinations
from typing import Callable
from typing import Type
from typing import Tuple
from collections.abc import Sequence
from numbers import Number
import traceback
    
from abc import ABC, abstractmethod
import numpy as np
        
class Function(ABC):
    @property
    @abstractmethod
    def forward_func(self) -> Callable:
        pass
    @property
    @abstractmethod
    def backward_func(self) -> Callable:
        pass

    def get_functions(self) -> Tuple[Callable, Callable]:
        return self.forward_func, self.backward_func

    def __call__(self, *inner) -> Expr_node: # simplifies the syntax for chaning of functions
        return Expr_node(self, inner)



class Sin(Function):
    forward_func = np.sin
    backward_func = np.cos

    #backward_func = lambda x: np.cos(x)
           #np.cos

class Tanh(Function):
    forward_func = np.tanh
    @property
    def backward_func(self):
        return lambda x: 1 - np.square(np.tanh(x)) if x.size == 1 else np.diag(1 - np.square(np.tanh(x)))

class Cos(Function):
    forward_func = np.cos
    @property
    def backward_func(self):
        return lambda x: -np.sin(x)


class Multiply(Function):

    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor

    def __init__(self,*, allow_arbitrary_many = False): # * makes allow_arbitary_many to keword only
        if not allow_arbitrary_many: # simple case two factors
            self.forward_func = np.multiply
            self.backward_func = lambda x, y: (y, x)
        else:
            self.forward_func = lambda *x: np.prod(np.vstack(x), axis=0)
            self.backward_func = lambda *x: (np.prod(np.vstack(values), axis=0) for values in combinations(x, len(x) - 1))

class Multiply_scalar(Function):

    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor

    def __init__(self,scalar):
        self.forward_func = lambda x: np.multiply(scalar, x)
        self.backward_func = lambda x: np.full_like(x, scalar)

class Add_scalar(Function):
    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor

    def __init__(self,scalar):
        self.forward_func = lambda x: np.add(scalar, x)
        self.backward_func = lambda x: np.full_like(x, 1)

class Scalar_pow(Function):
    #power function: a^b, a and b need to be scalars

    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor

    def __init__(self,scalar):
        self.forward_func = lambda x: x**scalar
        self.backward_func = lambda x: scalar*x**(scalar-1)



class Add(Function):

    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor

    def __init__(self,*, allow_arbitrary_many = False): # new is clalled before the constructer (before: "__init__")
        if not allow_arbitrary_many:
            self.forward_func = np.add
            self.backward_func = lambda x, y: (np.full_like(x, 1), np.full_like(y, 1)) #arrays with same shape of x and y, filled with 1
        else:
            self.forward_func = lambda *x: np.sum(np.vstack(x), axis=0)
            self.backward_func = lambda *x: (np.full_like(x[0], 1) for _ in range(len(x))) #backward_func = lambda *x: (np.ones(len(x[0]) if isinstance(x[0], Iterable) else 1) for _ in range(len(x))))



class Matrix_vector_product(Function):
    def matrix_vector_backwards(self, matrix, vector):
        c = np.zeros((len(matrix), len(matrix), len(vector)))
        for i in range(len(matrix)):
            c[i,i] = vector
        return (c,matrix)
    forward_func = np.matmul
    backward_func = matrix_vector_backwards

class Vector_vector_sum(Function):
    #vectors must have same length
    forward_func = np.add
    @property
    def backward_func(self):
        return lambda v1, v2: (np.eye(len(v1)), np.eye(len(v2)))


class Matrix_w_x_b(Function):
    @property
    def forward_func(self):
        return lambda w, x, b: w @ x + b
    def back_helper(self, w, x, b):
        dydw = np.zeros((len(w), len(x)*len(w)))
        for i in range(len(w)):
            dydw[i, i * len(x):(i + 1) * len(x)] = x
        dydx = w
        dydb = np.eye(len(b))
        return (dydw,dydx,dydb)
    backward_func = back_helper





class Log(Function):
    forward_func = np.log
    @property
    def backward_func(self):
        return lambda x: 1/x





class Expr_end_node():
    def __init__(self, value: Number | np.ndarray , grad_value: Number | np.ndarray  = 0):
        self.value = value
        self.grad_value = grad_value
        (filename,line_number,function_name,text) = traceback.extract_stack()[-2]
        iname = text[:text.find('=')].strip()
        self.instance = iname
        
class Expr_node():
    def __init__(self, func: Function, childs: Sequence[Expr_node | Expr_end_node] = []):
        #self.parents = parents #expr_node
        self.childs = childs  #expr_node
        #self.func = func #Function 
        # lieber so:
        self.forward_func, self.backward_func = func.get_functions()
        self.func_name = type(func).__name__
        # und dann self.func entfernen
        (filename,line_number,function_name,text) = traceback.extract_stack()[-2]
        iname = text[:text.find('=')].strip()
        self.instance = iname
