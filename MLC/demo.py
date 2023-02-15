import mlc
import numpy as np
import torch
import torch.nn as nn
import demo.model.resnet as Resnet
from tvm import relax
import tvm

# class Demo(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.w = nn.Parameter(torch.rand((900, 128)))
#         self.b = nn.Parameter(torch.rand((128,)))
#         self.bn = nn.BatchNorm2d(1, momentum=0)
#         self.conv2d = nn.Conv2d(1, 1, 3, 1)
#         self.relu = nn.ReLU()
#         self.linear = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = self.conv2d(x)
#         x = self.bn(x)
#         x = torch.relu(x)
#         x = x.view([1, -1])
#         x = torch.matmul(x, self.w)
#         x = torch.add(x, self.b)
#         x = self.relu(x)
#         x = self.linear(x)
#         return x
# resnet = Demo()
resnet = Resnet.resnet18()

hw = 384
x_np = np.random.rand(1, 1, hw, hw).astype('float32')
x_torch = torch.from_numpy(x_np)
# x_nd = tvm.nd.array(x_np, tvm.cuda(0))
x_nd = tvm.nd.array(x_np, tvm.cpu(0))

fx_module = torch.fx.symbolic_trace(resnet)

resnet_fx_module = mlc.from_fx(fx_module, [(1, 1, hw, hw)])
resnet_fx_module.show()

resnet_fused = mlc.FuseDenseAddPass()(resnet_fx_module)
# resnet_fused.show()

lowresnet = mlc.LowerToTensorIRPass()(resnet_fused)
lowresnet.show()


DemoModelFinal = relax.transform.FuseTIR()(lowresnet)
# DemoModelFinal.show()

# tunedResnet = mlc.mlc_tune_tir(DemoModelFinal, "cuda --max_threads_per_block=1024 --max_shared_memory_per_block=49152", max_trials_global=64, num_trials_per_iter=64, compile_tir_target='cuda')

# ex = relax.vm.build(tunedResnet, 'cuda')
# vm = relax.VirtualMachine(ex, tvm.cuda(0))
ex = relax.vm.build(DemoModelFinal, 'llvm')
vm = relax.VirtualMachine(ex, tvm.cpu(0))

res_nd = vm['main'](x_nd)
res_torch = resnet(x_torch)

np.testing.assert_allclose(res_nd.numpy(), res_torch.detach().numpy(), rtol=1e-5)

