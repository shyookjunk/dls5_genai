import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self,input_size=1,output_size=1):
    super().__init__()
    self.linear=nn.Linear(input_size,output_size)

  def forward(self,x):
    y=self.linear(x)
    return y
  
x=torch.rand(100,1)
y=5+2*x+torch.rand(100,1)

lr=0.1
iters=100

model=Model()
optimizaer=torch.optim.SGD(model.parameters(),lr=lr)

for i in range(iters):
  y_hat=model(x)
  loss=nn.functional.mse_loss(y,y_hat)

  loss.backward()
  optimizaer.step()   # update parameters
  optimizaer.zero_grad()

params=[]
for param in model.parameters():
  print(param)
  params.append(param)

import matplotlib.pyplot as plt
xx=torch.linspace(-0.05,1.05,3)
xx=xx.reshape((3,1))
#yy=params[0]*x+params[1]
yy=model(xx)
plt.scatter(x.data,y.data)
plt.plot(xx.data,yy.data,'r-',lw=2)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()