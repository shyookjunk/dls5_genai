import torch
import numpy as np

def rosenbrock(x0,x1):
  y=100*(x1-x0**2)**2+(x0-1)**2
  return y

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)
x00=[]
x11=[]

y=rosenbrock(x0,x1)
# y.backward()
# print(x0.grad, x1.grad)

iters=10000
lr=1e-3
for i in range(iters):
  #if i%10==0:
  x00.append(x0.item())
  x11.append(x1.item())
  if i%1000==0:
    print(f"iter {i}: x0={x0.item()}, x1={x1.item()}, y={y.item()}") 
  y=rosenbrock(x0,x1)
  y.backward()

  x0.data -= lr * x0.grad.data
  x1.data -= lr * x1.grad.data
  x0.grad.zero_()
  x1.grad.zero_()

print(f"Final: x0={x0.item()}, x1={x1.item()}, y={y.item()}")

# Plotting

x=np.linspace(-2,2,201)
y=np.linspace(-1,3,201)
epsilon=1e-8
X,Y=np.meshgrid(x,y)
Z=rosenbrock(X,Y)
Z=np.log(Z+epsilon)
# print(np.min(Z))
# print(np.max(Z))
levels=np.linspace(np.min(Z),np.max(Z),17)

import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
ax.contour(X,Y,Z,levels=levels)
#ax.contour(X,Y,Z)
ax.plot(x00,x11,'ro-')
plt.show()
