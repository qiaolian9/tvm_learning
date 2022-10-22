# improt package
import tvm
from tvm import te # te stands for tensor expression
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
try:
    import torch 
    print("\"import torch\" success...")
except:
    pass

def get_abc(shape, constructor = None):
    """ return random a, b and empty c with the same shape
    """
    np.random.seed(0)
    a = np.random.normal(size = shape).astype(np.float32)
    b = np.random.normal(size = shape).astype(np.float32)
    c = np.empty(shape = shape, dtype = np.float32)
    if constructor is not None:
        a, b, c = [constructor(i) for i in (a, b, c)]
    return a, b, c

# implemention with numpy
n = 100
a, b, c = get_abc(n, None)
c = a + b
def vector_add_normal(n, a, b):
    d = np.empty_like(a)
    for i in range(n):
        d[i] = a[i] + b[i]
    return d
d = vector_add_normal(n, a, b)
np.testing.assert_equal(c, d)

# defining the tvm computation
def vector_add(n):
    """TVM expressoion for vector add"""
    A = te.placeholder((n,), name = 'a')
    B = te.placeholder((n,), name = 'b')
    C = te.compute(A.shape, lambda i : A[i] + B[i], name = 'c')
    return A, B, C

A, B, C = vector_add(n)
print(type(A), type(C))
print(A.dtype,  A.shape)
print(type(A.op), type(C.op))
print(A.op.__class__.__bases__[0])

# create a schedule: execute plan
S = te.create_schedule(C.op)