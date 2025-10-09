import torch
import torch.nn as nn

class test_cls(nn.Module):
  def __init__(self):
    super().__init__()
    a=3.7
    b=2.6

  def forward(self,x):
    print("x=",x)

tst=test_cls()
tst(3)