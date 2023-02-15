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

**Demo**: resnet18(./MLC/demo/demo2.ipynb) 

**Including op**：Conv2d/BN/Pool/Linear/Add/Matmmul/Relu/Reshape/View etc.(./MLC/mlc/mlc.py)

