import numpy as np
import matplotlib.pyplot as plt
import norm_dist as nd

mus=np.array([[2.0, 54.50],
                [4.3, 80.0]])
covs=np.array([[[0.07, 0.44],
                  [0.44, 33.7]],
                 [[0.17, 0.94],
                  [0.94, 36.00 ]]])
phis=np.array([0.35, 0.65])

def sample():
  z=np.random.choice(2, p=phis)
  mu,cov=mus[z],covs[z]
  x=np.random.multivariate_normal(mu, cov)
  return x

def gmm(x,phis,mus,covs):
  k=len(phis)
  y=0
  for i in range(k):
    phi, mu, cov=phis[i], mus[i], covs[i]
    y+=phi*nd.multivariate_normal(x, mu, cov)
  return y

xs=np.linspace(1,6,61)
ys=np.linspace(40,100,61)
X,Y=np.meshgrid(xs, ys)
Z=np.zeros_like(X)
for i in range(X.shape[0]):
  for j in range(X.shape[1]):
    x=np.array([X[i,j], Y[i,j]])
    Z[i,j]=gmm(x,phis,mus,covs)

fig=plt.figure()
ax1=fig.add_subplot(1,2,1,projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2=fig.add_subplot(1,2,2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.contourf(X, Y, Z, cmap=None)
plt.show()
# N=500
# xs=np.zeros((N,2))
# for i in range(N):
#   xs[i]=sample()

# plt.scatter(xs[:,0], xs[:,1], alpha=0.7)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()