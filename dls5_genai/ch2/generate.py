import os
import numpy as np
import matplotlib.pyplot as plt
import norm_dist as nd

path=os.path.join(os.path.dirname(__file__),"height.txt")

xs=np.loadtxt(path)
print(xs.shape)
mu=np.mean(xs)
sigma=np.std(xs)

print(mu)
print(sigma)

# x=np.linspace(150,190,100)
# y=nd.normal(x,mu,sigma)

samples=np.random.normal(mu,sigma,10000)

plt.hist(xs,bins='auto',density=True,alpha=0.7,label='original data')
plt.hist(samples,bins='auto',density=True,alpha=0.7,label='generated data')
plt.legend()
# plt.plot(x,y)
plt.xlabel("Height(cm)")
plt.ylabel('Probability density')
plt.show()
