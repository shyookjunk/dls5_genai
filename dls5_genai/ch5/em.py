import os
import numpy as np
import norm_dist as nd

def likelihood(xs,phis,mus,covs):
  epsilon=1e-9
  L=0.0
  N=xs.shape[0]
  for x in xs:
    y=nd.gmm(x,phis,mus,covs)
    L+=np.log(y+epsilon)
  return L/N

ifn=os.path.join(os.path.dirname(__file__),"old_faithful.txt")
xs=np.loadtxt(ifn)
print("xs.shape=",xs.shape)

# parameter initialization
phis=np.array([0.5,0.5])    # K=2 인 모델. number of Gaussian dist.
mus=np.array([[0.0,50.0],[0.0,100.0]])
covs=np.array([np.eye(2),np.eye(2)],float)

K=len(phis)
N=xs.shape[0]
MAX_ITER=100
THRESHOLD=1e-4

current_likelihood=likelihood(xs,phis,mus,covs)
print(f"current likelihood={current_likelihood:.3f}")

# E-스텝과 M-스텝.
for iter in range(MAX_ITER):
  # E=step
  qs=np.zeros((N,K),float)
  for n in range(N):
    x=xs[n]
    for k in range(K):
      phi,mu,cov=phis[k],mus[k],covs[k]
      qs[n,k]=phi*nd.multivariate_normal(x,mu,cov)
    qs[n]/=nd.gmm(x,phis,mus,covs)

  # M-step
  qs_sum=qs.sum(axis=0)
  for k in range(K):
    # update phis
    phis[k]=qs_sum[k]/N
    # update mus
    #c=np.zeros(2,float)
    c=0
    for n in range(N):
      c+=qs[n,k]*xs[n]
    mus[k]=c/qs_sum[k]
    # update covs
    # c=np.zeros([len(x),len(x)],float)
    c=0
    for n in range(N):
      z=xs[n]-mus[k]
      #z=z[:,np.newaxis]
      z=z.reshape(len(z),1)
      c+=qs[n,k]*z@z.T
    covs[k]=c/qs_sum[k]

  print(f"iter:{iter}: current_likelihood: {current_likelihood:.3f}")

  # check the threshold.
  next_likelihood=likelihood(xs,phis,mus,covs)
  diff=np.abs(next_likelihood-current_likelihood)
  if diff<THRESHOLD:
    break
  current_likelihood=next_likelihood


# visualize
import matplotlib.pyplot as plt

def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * nd.multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)

plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()