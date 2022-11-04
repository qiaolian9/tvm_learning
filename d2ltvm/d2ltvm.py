import sys

d2ltvm = sys.modules[__name__]

import tvm
from tvm import te
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
try:
    import torch
    print("successd...\n")
except:
    print("Error in torch!")
    exit(0)

def get_abc(shape, construct=None):
    """ return random a,b and empty c with the same shape
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype('float32')
    b = np.random.normal(size=shape).astype('float32')
    c = np.empty_like(a)

    if(construct):
        a, b, c = [construct(i) for i in (a, b, c)]
    return a, b, c

def vector_add(n):
    """return tvm op vector_add
    """
    A = te.placeholder((n,),name='a')
    B = te.placeholder((n,),name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i],name='c')
    return A, B, C

def image_processing(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def broadcast_add(shape1, shape2):
    ''' broadcast add between two 2-dimensional tensors
        shape1, shape2 : the shapes of the input tensors
    '''
    assert len(shape1) == 2 and len(shape2) == 2, \
        "broadcast tensors should both be 2-dimension"
    for i in range(len(shape1)):
        assert shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1, \
            "tensor shapes do not fit for broadcasting"
    A = te.placeholder(shape1, name='A')
    B = te.placeholder(shape2, name='B')
    m = shape1[0] if shape2[0] == 1 else shape2[0]
    n = shape1[1] if shape2[0] == 1 else shape2[1]
    f = lambda x, y: A[0 if shape1[0] == 1 else x, 0 if shape1[1] == 1 else y] + \
                B[0 if shape2[0] == 1 else x, 0 if shape2[1] == 1 else y]
    C = te.compute((m, n), f, name='C')
    return A, B, C

def get_broad_data(shape1, shape2, constructor=None):
    """ Return random tensors a, b
        and empty tensor c to store broadcast results between a and b
        shape1, shape2: shapes of input tensors
        constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype('float32')
    b = np.random.normal(size=shape2).astype('float32')
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0], 
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = np.empty(shape=out_shape).astype('float32')
    if constructor:
        a, b, c = [constructor(i) for i in (a, b, c)]
    return a, b, c

def matmul(n, m, l):
    """ Return the computing expression of matrix multiplication
        A : n x l matrix
        B : l x m matrix
        C : n x m matrix with C = A B
    """
    A = te.placeholder((n, l), name='A', dtype='float32')
    B = te.placeholder((l, m), name='B', dtype='float32')
    l = te.reduce_axis((0, l), name='k')
    C = te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    return A, B, C