import numpy as np
import matplotlib.pyplot as plt
import norm_dist as nd

mu=np.array([0.5,-0.2])
sigma=np.array([[2.0,0.3],[0.3,0.5]])

xrange=np.linspace(-5,5,101)
yrange=np.linspace(-5,5,101)
X,Y=np.meshgrid(xrange,yrange)
Z=np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x=np.array([X[i,j],Y[i,j]])
        Z[i,j]=nd.multivariate_normal(x,mu,sigma)

fig=plt.figure()
ax1=fig.add_subplot(1,2,1,projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X,Y,Z,cmap='viridis')

ax2=fig.add_subplot(1,2,2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.contour(X,Y,Z)

plt.show()