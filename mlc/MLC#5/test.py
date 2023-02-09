import torch
import torch.nn as nn

# torch version
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(128, 128))
    
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x

demo = MyModel()
print(demo)