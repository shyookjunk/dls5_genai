import torch
import torch.nn.functional as F

torch.manual_seed(0)
x=torch.rand(100,1)
y=2*x+5+torch.rand(100,1)

W=torch.zeros((1,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def predict(x):
  y=x@W+b
  return y

def mse(x0,x):
  diff=x0-x
  N=len(diff)
  return torch.sum(diff**2)/N

lr=0.1
iters=100

for i in range(iters):
  y_hat=predict(x)
  #loss=mse(y,y_hat)
  loss=F.mse_loss(y,y_hat)

  loss.backward()

  W.data-=lr*W.grad.data
  b.data-=lr*b.grad.data

  W.grad.zero_()
  b.grad.zero_()

  if i%10==0:
    print(loss.item())

print(loss.item())
print("------------------------------")
print("W=",W.item())
print("b=",b.item())


import matplotlib.pyplot as plt
import numpy as np
xx=torch.linspace(-0.05,1.05,4)
xx=xx.reshape((4,1))
yy=predict(xx)
plt.plot(x.data,y.data,'o',mfc='none',mec='blue',lw=2,ms=8)
plt.plot(xx.data,yy.data,'r-',lw=2)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
