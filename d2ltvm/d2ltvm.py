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

# 3_1_Broadcast
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

# 3_2_matmul
def matmul(n, m, l):
    """ Return the computing expression of matrix multiplication
        A : n x l matrix
        B : l x m matrix
        C : n x m matrix with C = A B
    """
    A = te.placeholder((n, l), name='A', dtype='float32')
    B = te.placeholder((l, m), name='B', dtype='float32')
    k = te.reduce_axis((0, l), name='k')
    C = te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    return A, B, C

# 3_3_Convolution
def padding(X, ph, pw, val=0):
    """ Pad X with the given value in 2-D
        ph, pw : height and width padding
        val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute((*X.shape[0:-2], nh + 2 * ph, nw + 2 * pw), 
                      lambda *i: te.if_then_else(
                          te.any(i[-2] < ph, i[-2] >= ph + nh, i[-1] < pw, i[-1] >= pw + nw), 
                          val, X[i[:-2] + (i[-2] - ph, i[-1] - pw)]),
                     name='PaddedX')

def conv_out_size(n, k, p, s):
    """ Compute the output size by given input size n (width or height),
        kernel size k, padding p, and stride s
        Return output size (width or height)
    """
    return (n + 2 * p - k) // s + 1

def conv(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """ Convolution
        oc, ic : output and input channels
        nh, nw : input width and height
        kh, kw : kernel width and height
        ph, pw : height and width padding sizes, default 0
        sh, sw : height and width strides, default 1
    """
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, nh, nw), name='K')
    
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute((oc, oh, ow), 
            lambda c, i, j: te.sum(PaddedX[ric, i * sh + rkh, j * sw + rkw] * K[c, ric, rkh, rkw], 
                    axis=[ric, rkh, rkw]), name='Y')
    return X, K, Y, PaddedX

def get_conv_data(oc, ic, n, k, p, s=1, constructor=None, conv_type='direct'):
    """ Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
        tensor with the shapes specified by input arguments.
        oc, ic : output and input channels
        n : input width and height
        k : kernel width and height
        p : padding size, default 0
        s : stride, default 1
        constructor : user-defined tensor constructor
        conv_type: either direct or depthwise, default direct
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    ic_weight = ic
    if conv_type == 'depthwise':
        ic_weight = 1
    weight = np.random.normal(size=(oc, ic_weight, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty(shape=(oc, on, on)).astype('float32')
    if constructor is not None:
        data, weight, out = [constructor(i) for i in (data, weight, out)]
    
    return data, weight, out

def get_conv_data_torch(oc, ic, n, k, p, s, ctx='cpu', conv_type='direct'):
    device = torch.device(ctx)
    data, weight, _ = get_conv_data(oc, ic, n, k, p, s, constructor=lambda x: torch.tensor(x, device=device), conv_type=conv_type)
    data = data[None, ...]
    bias = torch.zeros(oc, device=device)
    return data, weight, bias

def conv_torch(data, weight, bias, p, s):
    return torch.nn.functional.conv2d(data, weight, bias=bias, stride=s, padding=p)

# 3_4_DepthwiseConvolution
def depthwise_conv(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """ Convolution
        ic : number of channels for both input and output
        nh, nw : input width and height
        kh, kw : kernel width and height
        ph, pw : height and width padding sizes, default 0
        sh, sw : height and width strides, default 1
    """
    # reduction axis
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((ic, 1, kh, kw), name='Y')
    PaddedX = padding(X, ph, pw) if ph | pw != 0 else X
    Y = te.compute((ic, oh, ow),
                  lambda c, i, j: te.sum(PaddedX[c, i * sh + rkh, j * sw + rkw] * K[c, 0, rkh, rkw],
                    axis=[rkh, rkw]), name='Y')
    return X, K, Y, PaddedX

def depthwise_conv_torch(data, weight, bias, p, s):
    return torch.nn.functional.conv2d(data, weight, bias, s, p, groups=data.shape[1])

# 3_5_Pooling
def pool(pool_type, c, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """ 2D pooling
        pool_type: pooling type, 'max' or 'avg'
        c : channels
        nh, nw : input width and height
        kh, kw : kernel width and height
        ph, pw : height and width padding sizes, default 0
        sh, sw : height and width strides, default 1
    """
    # reduction axis
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    X = te.placeholder((c, nh, nw), name='X')
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    
    if pool_type == 'max':
        PaddedX = padding(X, ph, pw, val=te.min_value(X.dtype)) if ph | pw != 0 else X
        Y = te.compute((c, oh, ow), \
                      lambda c, i, j: te.max(PaddedX[c, i * sh + rkh, j * sw + rkw],\
                        axis=[rkh, rkw]), tag='pool_max', name='PoolMax')
    elif pool_type == 'avg':
        PaddedX = padding(X, ph, pw, val=0) if ph | pw != 0 else X
        tSum = te.compute((c, oh, ow), \
                      lambda c, i, j: te.sum(PaddedX[c, i * sh + rkh, j * sw + rkw], \
                        axis=[rkh, rkw]), tag='pool_avg1', name='PoolSum')
        Y = te.compute((c, oh, ow), \
                      lambda c, i, j: tSum[c, i, j] / (kh * kw), \
                        tag='pool_avg2', name='PoolAvg')
    else:
        raise ValueError("pool type should be 'avg' or 'max'")
    return X, Y, PaddedX

def get_pool_data_torch(c, n, k, p, s, ctx='cpu'):
    device = torch.device(ctx)
    data, _, out = get_conv_data(c, c, n, k, p, s, lambda x: torch.tensor(x, device=device))
    data, out = data[None, ...], out[None, ...]
    return data, out

def pool_torch(pool_type, data, k, p, s):
    if pool_type == 'avg':
        torch.nn.functional.avg_pool2d(data, k, s, p)
    elif pool_type == 'max':
        torch.nn.functional.max_pool2d(data, k, s, p)
    else:
        raise ValueError("pool type should be 'avg' or 'max'")

# 3_6_BatchNormalization
from tvm import topi
def batch_norm(c, n, eps=1e-5):
    """ batch normalization
        c : channels
        N : input width and height
        eps : small positive value to prevent divide 0
    """
    X = te.placeholder((c, n, n), name='X')
    Mean = te.placeholder((c, 1, 1), name='Mean')
    Var = te.placeholder((c, 1, 1), name='Var')
    Gamma = te.placeholder((c, 1, 1), name='Gamma')
    Beta = te.placeholder((c, 1, 1), name='Beta')
    C1 = X - Mean
    C2 = topi.sqrt(Var + eps)
    Y = C1 / C2 * Gamma + Beta
    return X, Mean, Var, Gamma, Beta, Y

def get_bn_data(c, n, constructor=None):
    """ Return the batch norm data, mean, variance, gamma and beta tensors.
        Also return the empty tensor for output.
        c : channels
        n : input width and height
        constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(c, n, c)).astype('float32')
    mean = np.random.normal(size=(c, 1, 1)).astype('float32')
    
    var = np.random.normal(size=(c, 1, 1)).astype('float32')
    var = np.absolute(var)
    
    gamma = np.random.normal(size=(c, 1, 1)).astype('float32')
    beta = np.random.normal(size=(c, 1, 1)).astype('float32')
    out = np.empty(shape=(c, n, n)).astype('float32')
    
    if constructor is not None:
        data, mean, var, gamma, beta, out = \
            [constructor(x) for x in (data, mean, var, gamma, beta, out)]
    
    return data, mean, var, gamma, beta, out


def batch_norm_torch(data, mean, var, gamma, beta, eps=1e-5):
    return torch.nn.functional.batch_norm(data, mean, var, gamma, beta, eps=eps)

# 4_2_FunctionCallOverhead
def bench_workload(workload):
    """ Benchmark a workload
        workload: a method that accept a num_repeat argument
        and return its total execution time
    """
    workload(1) #warmup
    time = workload(1)
    if time > 1 :
        return time
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

# 4_3_VectorAdd
def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=None, figsize=(4.5, 3)):
    """Plot multiple lines"""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    X, Y = np.array(X), np.array(Y)
    if X.shape != Y.shape: X = [X] * len(Y)
    if not fmts: fmts = ['-'] * len(X)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()

def plot_gflops(sizes, gflops, legend, xlabel='size'):
    plot(sizes, gflops, xlabel=xlabel, ylabel='GFLOPS',
            xscale='log', yscale='log', legend=legend, fmts=['--']*(len(gflops)-1)+['-'])

def bench_vector_add_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 0)
        a, b, c = get_abc(n, lambda x: tvm.nd.array(x, ctx=ctx))
        times.append(bench_workload(workload))
    return sizes / 1e9 / np.array(times)

# 4_4_Broadcast_Add
def bench_broad_add_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(n)
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 0)
        a, b, c = get_broad_data((n, 1), (n, n), lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    return sizes * sizes / 1e9 / np.array(times)

# 4_5_MatrixMultiplication
def np_matmul_timer(n):
    timer = timeit.Timer(setup='import numpy as np\n'
                'import d2ltvm\n'
                'a, b, c = d2ltvm.get_abc(%s)' % str((n, n)),
                stmt='np.dot(a, b, out=c)')
    return timer.timeit

def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(n)
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 0)
        a, b, c = get_abc((n, n), lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    
    return 2 * sizes ** 3 / 1e9 / np.array(times)
