import norm_dist as nd
import numpy as np

x=np.array([[0],[0]])
mu=np.array([[1],[2]])
sigma=np.array([[1,0],[0,1]])

y=nd.multivariate_normal(x,mu,sigma)

print("y=")
print(y)