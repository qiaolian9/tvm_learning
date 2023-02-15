# MLC learning
```
.
├── d2ltvm_tutorial
│   ├── ch1_getting_started
│   ├── ch2_expressions
│   ├── ch3_Common_Operators
│   ├── ch4_Operator_Optimization
│   ├── d2ltvm
│   ├── README.md
│   └── start.sh
├── MLC
│   ├── demo
│   ├── demo.py
│   ├── __init__.py
│   ├── mlc
│   ├── MLC#2
│   ├── MLC#3
│   ├── MLC#4
│   ├── MLC#5
│   ├── MLC#6
│   ├── MLC#7
│   └── README.md
└── README.md
```
## source
d2ltvm_tutorial: <https://github.com/d2l-ai/d2l-tvm> \
mlc: <https://mlc.ai/zh/>

## MLC
detail in ./MLC/README.md

**Process**: torch.nn.Module ---> relax.function ---> FusedOp ---> low level TensorIR ---> tuned mod ---> ex 
```python
import mlc
import tvm
resnet = resnet18() # nn.Module

resnet_fx_module = mlc.from_fx(resnet, [(1, 1, 384, 384)]) # relax.function

resnet_fused_module = mlc.FuseDenseAddPass()(resnet_fx_module) # graph op & op fused

lowresnet = mlc.LowerToTensorIRPass()(resnet_fused_module)  # Low level TensorIR
resnetFinal = relax.transform.FuseTIR()(lowresnet)

tunedResnet = mlc.mlc_tune_tir(DemoModelFinal, "cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152", max_trials_global=64, num_trials_per_iter=64, compile_tir_target='cuda')     # tuned op

ex = relax.vm.build(tunedResnet, target='cuda')
vm = relax.VirtualMachine(ex, tvm.cuda(0))   #relax virtualmachine
```


**Demo**: resnet18(./MLC/demo/demo2.ipynb) 

**Including op**：Conv2d/BN/Pool/Linear/Add/Matmmul/Relu/Reshape/View etc.(./MLC/mlc/mlc.py)

