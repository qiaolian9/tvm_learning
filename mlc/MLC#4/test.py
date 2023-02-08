from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
from tvm.script import tir as T
from tvm.script import relax as R
from tvm import IRModule

@tvm.script.ir_module
class MyModuleMixture():
    @T.prim_func
    def linear0(x: T.Buffer[(1, 784), 'float32'],
                w0: T.Buffer[(128, 784), 'float32'],
                b0: T.Buffer[(128,), 'float32'],
                z: T.Buffer[(1, 128), 'float32']):
        T.func_attr({"global_symbol": "linear0", 'tir.noalias': True})
        lv0 = T.alloc_buffer((1, 128), dtype='float32')
        for i, j, k in T.grid(1, 128, 784):
            with T.block('Y'):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    lv0[vi, vj] = T.float32(0)
                lv0[vi, vj] = lv0[vi, vj] + x[vi, vk] * w0[vj, vk]
        
        for i, j in T.grid(1, 128):
            with T.block('C'):
                vi, vj = T.axis.remap('SS', [i, j])
                z[vi, vj] = lv0[vi, vj] + b0[vj]
    
    @R.function
    def main(x: R.Tensor((1, 128), 'float32'),
             w0: R.Tensor((128, 784), 'float32'),
             b0: R.Tensor((128,), 'float32'),
             w1: R.Tensor((10, 128), 'float32'),
             b1: R.Tensor((10,), 'float32')):
        with R.dataflow():
            lv0 = R.call_tir(linear0, (x, w0, b0), relax.TensorStructInfo((1, 128), 'float32'))
            # lv1 = R.call_tir('env.relu', (lv0,), relax.TensorStructInfo((1, 128), 'float32'))
            # out = R.call_tir('env.linear', (lv1, w1, b1), relax.TensorStructInfo((1, 10), 'float32'))
            out = lv0
            R.output(out)
        
        return out

print(MyModuleMixture)