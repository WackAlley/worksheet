from __future__ import annotations
from itertools import combinations
from typing import Callable
from typing import Type
from typing import Tuple
from collections.abc import Sequence
from numbers import Number
#from icecream import ic
#from nodes import *
    
    
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

class Tanh(Function):
    forward_func = np.tanh
    backward_func = lambda x: 1 - np.square(tanh(x))

class Cos(Function):
    forward_func = np.cos
    backward_func = lambda x: -np.sin(x)
          

class Multiply(Function):
    
    forward_func = lambda: None # to be overwitten in constructor
    backward_func = lambda: None # to be overwitten in constructor
    
    def __init__(self,*, allow_arbitrary_many = False): # * makes allow_arbitary_many to keword only
        if not allow_arbitrary_many: # simple case two factors
            self.forward_func = np.multiply
            self.backward_func = lambda x, y: (y, x)
        else:
            forward_func = lambda *x: np.prod(np.vstack(x), axis=0)
            self.backward_func = lambda *x: (np.prod(np.vstack(values), axis=0) for values in combinations(x, len(x) - 1))
                   
            
        
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
        return (c, matrix) 
    forward_func = np.matmul
    backward_func = matrix_vector_backwards
    
class Vector_vector_sum(Function):
    #vectors must have same length
    forward_func = np.add
    backward_func = lambda v1, v2: (np.eye(len(v1)), np.eye(len(v2)))





class Expr_end_node():
    def __init__(self, value: Number | np.ndarray , grad_value: Number | np.ndarray  = 0):
        self.value = value
        self.grad_value = grad_value
        
class Expr_node():
    def __init__(self, func: Function, childs: Sequence[Expr_node | Expr_end_node] = []):
        #self.parants = parants #expr_node
        self.childs = childs  #expr_node
        #self.func = func #Function 
        # lieber so:
        self.forward_func, self.backward_func = func.get_functions()
        self.func_name = type(func).__name__
        # und dann self.func entfernen
