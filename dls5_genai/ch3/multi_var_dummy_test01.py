import numpy as np

N=10000
D=2

xs=np.random.rand(N,D)

mu=np.sum(xs,axis=0)
mu/=N

cov=0

for n in range(N):
    x=xs[n]
    z=x-mu
    z=z[:,np.newaxis]
    cov+=z@z.T

cov/=N

print("mu=",mu)
print("cov=")
print(cov)