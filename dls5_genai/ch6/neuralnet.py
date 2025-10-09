import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self,input_size=1,hidden_size=10,output_size=1):
    super().__init__()
    self.linear1=nn.Linear(input_size,hidden_size)
    self.linear2=nn.Linear(hidden_size,output_size)

  def forward(self,x):
    y=self.linear1(x)
    y=F.sigmoid(y)
    y=self.linear2(y)
    return y

torch.manual_seed(0)
x=torch.rand(100,1)
y=torch.sin(2*torch.pi*x)+torch.rand(100,1)

lr=0.2
iters=10000

model=Model()
optimizer=torch.optim.SGD(model.parameters(),lr)

for i in range(iters):
  y_pred=model(x)
  loss=F.mse_loss(y,y_pred)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  if i%1000==0:
    print(loss.item())

print(loss.item())

import matplotlib.pyplot as plt
x_pred=torch.linspace(0,1.0,100)
x_pred=x_pred.reshape((100,1))
y_pred=model(x_pred)
plt.scatter(x,y)
plt.plot(x_pred.data,y_pred.data,'r-',lw=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
