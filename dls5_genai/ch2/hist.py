import os
import numpy as np
import matplotlib.pyplot as plt
import norm_dist as nd

xs=np.loadtxt("height.txt")
print(xs.shape)
mu=np.mean(xs)
sigma=np.std(xs)

print(mu)
print(sigma)

x=np.linspace(150,190,100)
y=nd.normal(x,mu,sigma)

plt.hist(xs,bins='auto',density=True)
plt.plot(x,y)
plt.xlabel("Height(cm)")
plt.ylabel('Probability density')
plt.show()
