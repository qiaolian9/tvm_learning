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
    c = np.empty(shape=shape)

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
