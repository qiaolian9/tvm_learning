from __future__ import annotations
import numpy as np
import tvm
from torch import fx
from tvm import te, relax, IRModule
from tvm import meta_schedule as ms
from typing import List, Optional, Tuple, Union
from tvm import topi
from tvm.script import relax as R
from tvm.script import tir as T
import torch.nn as nn
import torch
import sys
import builtins
mlc = sys.modules[__name__]
# tvm.ir.container.Array().
# ---------------------- te ---------------------------------------#
def mean_te(x, axis):
    shape = x.shape
    reduce_axis = []
    mean_ = 1
    for i in axis:
        reduce_axis.append(te.reduce_axis((0, shape[int(i)])))
        mean_ *= shape[int(i)]
    mean_ = te.const(int(mean_))
    sum_ = te.compute((int(shape[1]),), lambda i: te.sum(x[reduce_axis[0],i,reduce_axis[1],reduce_axis[2]], axis=reduce_axis), name='sum_out')
    out = te.compute(sum_.shape, lambda *i: sum_(*i) / mean_, name='mean_out')
    return out

def var_te(x, axis):
    shape = x.shape
    reduce_axis = []
    n_ = 1
    for i in axis:
        reduce_axis.append(te.reduce_axis((0, shape[int(i)])))
        n_ *= shape[int(i)]
    n_ = te.const(int(n_))
    mean_ = mean_te(x, axis)
    sub_ = te.compute(x.shape, lambda n, c, h, w: te.power(x[n, c, h, w] - mean_[c], 2), name='sub_out')
    sum_ = te.compute(mean_.shape, lambda i: te.sum(sub_[reduce_axis[0],i,reduce_axis[1],reduce_axis[2]], axis=reduce_axis), name='sum_out')
    var_out = te.compute(sum_.shape, lambda i: sum_[i] / n_, name='var_out')
    return var_out

# ---------------------- 构造映射函数 ---------------------------------------#
# torch graph/nn.Module --> IR high level relax.function
def map_params(param: nn.Parameter):
    return relax.const(param.data.cpu().numpy(), dtype='float32')

def fetch_attr(fx_mod, target: str):
    # 获取mod属性
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

# call_function
def map_matmul(bb: relax.BlockBuilder, node_map, node):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    w = node_map[node.args[1]]
    return bb.emit(relax.op.matmul(x, w))

def map_relu(bb: relax.BlockBuilder, node_map, node):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    return bb.emit(relax.op.nn.relu(x))

def map_add(bb: relax.BlockBuilder, node_map, node):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    b = node_map[node.args[1]]
    if isinstance(b.struct_info, relax.TupleStructInfo):
       b = b[0]
    return bb.emit(relax.op.add(x, b))

def map_flatten(bb: relax.BlockBuilder, node_map, node):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    dims = node.args[1]
    shape = np.array([int(i) for i in x.struct_info.shape])
    new_shape = shape[:dims]
    new_shape = np.append(new_shape, shape[dims:].prod())
    
    return bb.emit(relax.op.reshape(x, tuple(new_shape)))

def map_other_op(bb: relax.BlockBuilder, node_map, node):
    pass

# call_module
def map_nn_relu(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    return bb.emit(relax.op.nn.relu(x))

def map_nn_linear(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    w = map_params(nn_module.weight)
    bias = map_params(nn_module.bias)
    return bb.emit(relax.op.linear(x, w, bias))

def map_nn_conv2d(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    kernel = map_params(nn_module.weight)
    bias = None
    if nn_module.bias is not None:
        bias = map_params(nn_module.bias)
    strides = nn_module.stride
    padding = nn_module.padding
    dilation = nn_module.dilation
    conv_out = bb.emit(relax.op.nn.conv2d(x, kernel, strides, padding, dilation))
    if bias:
        return bb.emit(relax.op.add(conv_out, bias))
    return conv_out

def map_nn_BatchNorm2d(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    affine = nn_module.affine
    eps = nn_module.eps

    # moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
    # moving_var = moving_var * momentum + data_var * (1 - momentum)
    data_mean = bb.emit(relax.op.mean(x, [0, 2, 3]))
    data_var = bb.emit(relax.op.variance(x, [0, 2,3]))

    # if node not in node_map.keys(): 
    #     moving_mean_start = map_params(nn_module.running_mean)
    #     moving_var_start = map_params(nn_module.running_var)
    #     moving_mean = relax.Var(node.target+'_moving_mean', R.Tensor(nn_module.running_mean.shape, 'float32'))
    #     moving_var = relax.Var(node.target+'_moving_var', R.Tensor(nn_module.running_var.shape, 'float32'))

    # momentum = nn_module.momentum
    moving_mean = data_mean
    moving_var = data_var
    
    
    if affine:
        gamme = map_params(nn_module.weight)
        beta = map_params(nn_module.bias)
    bn = relax.op.nn.batch_norm(x, gamme, beta, moving_mean, moving_var, epsilon=eps, axis=1)
    return bb.emit(bn)

def map_nn_maxpool(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    pool_size = nn_module.kernel_size
    strides = nn_module.stride
    padding = nn_module.padding
    return bb.emit(relax.op.nn.max_pool2d(x, pool_size, strides, padding))

def map_nn_avgpool(bb: relax.BlockBuilder, node_map, node, nn_module):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    output_size = nn_module.output_size
    return bb.emit(relax.op.nn.adaptive_avg_pool2d(x, output_size))

# call_method
def map_view(bb: relax.BlockBuilder, node_map, node):
    x = node_map[node.args[0]]
    if isinstance(x.struct_info, relax.TupleStructInfo):
        x = x[0]
    shape = node.args[1]
    return bb.emit(relax.op.reshape(x, shape))

call_map_function = {
    torch.matmul: map_matmul,
    torch.add: map_add,
    torch.relu: map_relu,
    torch.flatten: map_flatten,
    # torch.Tensor.__add__: map_add
}

call_map_module = {
    torch.nn.Linear: map_nn_linear,
    torch.nn.ReLU: map_nn_relu,
    torch.nn.Conv2d: map_nn_conv2d,
    torch.nn.BatchNorm2d: map_nn_BatchNorm2d,
    torch.nn.MaxPool2d: map_nn_maxpool,
    torch.nn.AdaptiveAvgPool2d: map_nn_avgpool,
}

call_map_method = {
    'view': map_view
}

def from_fx(fx_module: Union[torch.fx.GrapgModule, nn.Module], input_shapes, 
            call_map_function=call_map_function, 
            call_map_module=call_map_module, 
            call_map_method=call_map_method):
    '''
        torch.fx.graph ---> high level relax function 
    '''
    if isinstance(fx_module, nn.Module):
        fx_module = fx.symbolic_trace(fx_module)
    input_index = 0
    node_map = {}
    named_modules = dict(fx_module.named_modules())

    bb = relax.BlockBuilder()
    fn_inputs = []
    fn_output = None
    with bb.function('main'):
        with bb.dataflow():
            for node in fx_module.graph.nodes:
                if node.op == 'placeholder':
                    input_shape = input_shapes[input_index]
                    input_index = input_index + 1
                    fn_input = relax.Var(node.target, R.Tensor(input_shape, 'float32'))
                    fn_inputs.append(fn_input)
                    node_map[node] = fn_input
                elif node.op == 'get_attr':
                    node_map[node] = map_params(fetch_attr(fx_module, node.target))
                elif node.op == 'call_function':
                    if node.target in call_map_function.keys():
                        node_map[node] = call_map_function[node.target](bb, node_map, node)
                    else:  # 内置 "+" 运算符定义错误
                        node_map[node] = call_map_function[torch.add](bb, node_map, node)
                # --------------------- add call_method ------------------------#
                elif node.op == 'call_method':
                    node_map[node] = call_map_method[node.target](bb, node_map, node)
                # --------------------------------------------------------------#
                elif node.op == 'call_module':
                    nn_module = named_modules[node.target]
                    # node_map[node] = call_map_module[nn_module](bb, node_map, node, nn_module)
                    node_map[node] = call_map_module[type(nn_module)](bb, node_map, node, nn_module)
                elif node.op == 'output':
                    output = node_map[node.args[0]]
                    if isinstance(output.struct_info, relax.TupleStructInfo):
                        output = output[0]
                    if fn_output is not None:
                        raise Warning("error")
                    fn_output = bb.emit_output(output)
        bb.emit_func_output(fn_output, fn_inputs)
    return bb.get()
# ------------------------------------------------------------- #

# ---------------------- 图优化 ---------------------------------------#
# 算子融合 high level relax function ---> high level relax function
# eg: matmul & add
def create_fuse_dense_add(call: relax.Call, value: relax.Call, func=None, fn_name=None):
    b = call.args[1]
    x = value.args[0]
    w = value.args[1]
    T_ = False

    if not isinstance(w, relax.Constant):
        value = func(w)
        # if value.op == tvm.ir.Op.get('relax.permute_dims')
        w = func(w).args[0]
        T_ = True

    params_x = relax.Var('x', x.struct_info)
    params_w = relax.Var('w', w.struct_info)
    params_b = relax.Var('b', b.struct_info)

    bb = relax.BlockBuilder()
    with bb.function(fn_name, [params_x, params_w, params_b]):
        with bb.dataflow():
            if T_:
                lv0 = bb.emit(relax.op.linear(params_x, params_w))
            else:
                lv0 = bb.emit(relax.op.matmul(params_x, params_w))
            
            gv = bb.emit_output(bb.emit(relax.op.add(lv0, params_b)))
        bb.emit_func_output(gv)
    
    fused_fn = bb.get()[fn_name].with_attr('Primitive', 1)
    return fused_fn, x, w, b

@relax.expr_functor.mutator
class DenseAddFusor(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule] = None):
        super().__init__()
        self.mod_ = mod
        self.add_op = tvm.ir.Op.get('relax.add')
        self.permute_op = tvm.ir.Op.get('relax.permute_dims')
        self.matmul_op = tvm.ir.Op.get('relax.matmul')
        self.counter = 0
    
    def transform(self):
        # relax function 
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is not None and "Primitive" in func.attrs and func.attrs['Primitive'] != 0:
                continue

            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)
        return self.builder_.get()
    
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op
        
        if not match_call(call, self.add_op):
            return call
        
        value = self.lookup_binding(call.args[0])
        if value is None or not match_call(value, self.matmul_op):
            return call
        
        fn_name = "fused_dense_add%d" % (self.counter)
        self.counter += 1
        fused_fn, x, w, b = create_fuse_dense_add(call, value, self.lookup_binding, fn_name)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        return relax.Call(global_var, [x, w, b], None, None)

@tvm.ir.transform.module_pass(opt_level=2, name="DeseAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return DenseAddFusor(mod).transform()
# ------------------------------------------------------------- #

# ---------------------- 映射至TensorIR Call ---------------------------------------#
# high level relax function ---> Tensor IR function
def map_mean_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    attrs = call.attrs
    x = call.args[0]
    return bb.call_te(mean_te, x, attrs.axis)
    
def map_var_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    attrs = call.attrs
    x = call.args[0]
    return bb.call_te(var_te, x, attrs.axis)

def map_add_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x, b = call.args
    return bb.call_te(topi.add, x, b)
    
def map_matmul_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x, w = call.args
    if isinstance(w, relax.expr.DataflowVar) and (func(w).op == tvm.ir.Op.get('relax.permute_dims')):
        w = func(w).args[0]
        return bb.call_te(topi.nn.dense, x, w)
    return bb.call_te(topi.nn.matmul, x, w)

def map_relu_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x = call.args[0]
    return bb.call_te(topi.nn.relu, x)

def map_conv_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x, k = call.args
    attrs = call.attrs
    return bb.call_te(topi.nn.conv2d, x, k, attrs.strides, attrs.padding, attrs.dilation)

def map_reshape_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x, shape = call.args
    return bb.call_te(topi.reshape, x, shape)

def map_BatchNorm2d_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x, gamma, beta, moving_mean, moving_var = call.args[:5]
    attrs = call.attrs
    return bb.call_te(topi.nn.batch_norm, x, gamma, beta, moving_mean, moving_var, attrs.axis, attrs.epsilon, attrs.center, attrs.scale)

def map_maxpool2d_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.pool2d, x, attrs.pool_size, attrs.strides, attrs.dilation, attrs.padding, 'max', attrs.ceil_mode, attrs.layout)

def map_adaptiveAvgPool2d_te(bb: relax.BlockBuilder, call: relax.Call, func=None):
    x = call.args[0]
    attrs = call.attrs
    return bb.call_te(topi.nn.adaptive_pool, x, attrs.output_size, 'avg', attrs.layout)

op_map = {
    'relax.variance': map_var_te,
    'relax.mean': map_mean_te,
    'relax.matmul': map_matmul_te,
    'relax.add': map_add_te,
    'relax.nn.relu': map_relu_te,
    'relax.nn.conv2d': map_conv_te,
    'relax.reshape': map_reshape_te,
    'relax.nn.batch_norm': map_BatchNorm2d_te,
    'relax.nn.max_pool2d': map_maxpool2d_te,
    'relax.nn.adaptive_avg_pool2d': map_adaptiveAvgPool2d_te,
}


@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: Optional[IRModule], op_map=op_map):
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }
    
    def visit_call_(self, call: relax.Call):
        call = self.visit_expr_post_order(call)
            
        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call, self.lookup_binding)
        return call
    
    def transform(self):
        for global_val, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_fn = self.visit_expr(func)
            updated_fn = relax.analysis.remove_all_unused(updated_fn)
            self.builder_.update_func(global_val, updated_fn)

        return self.builder_.get()

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()


# ------------------------------------------------------------- #

# ---------------------- 算子优化 ---------------------------------------#
# 只支持一个main的IRModule优化
def mlc_tune_tir(Model: IRModule, target='cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152', 
                work_dir="./tune_tmp/",
                task_name='main',
                max_trials_global=64,
                num_trials_per_iter=64,
                compile_tir_target='cuda'):
    print("target: %s; compile_tie_target: %s" % (target, compile_tir_target))
    fn_names = [x.name_hint for x in Model.functions]
    fn_names.remove('main')
    print(len(fn_names))
    for i, fn_name in enumerate(fn_names):
        print("op-%d : %s" %(i, fn_name))
        mod_ = tvm.IRModule.from_expr(Model[fn_name].with_attr("global_symbol", 'main'))

        tuned_record = ms.tune_tir(mod_, target=target,
                            work_dir=work_dir,
                            task_name=task_name,
                            max_trials_global=max_trials_global,
                            num_trials_per_iter=num_trials_per_iter)
        tuned_sch = ms.tir_integration.compile_tir(tuned_record, mod_, target=compile_tir_target)
        new_func = tuned_sch.mod['main'].with_attr("global_symbol", fn_name)
        gv = Model.get_global_var(fn_name)
        Model.update_func(gv, new_func)
    
    return Model