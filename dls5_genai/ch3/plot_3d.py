import numpy as np
import matplotlib.pyplot as plt

x=np.array([[-2,-1,0,1,2],
            [-2,-1,0,1,2],
            [-2,-1,0,1,2],
            [-2,-1,0,1,2],
            [-2,-1,0,1,2]])
y=x.T

z=x**2+y**2

ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

xrange=np.linspace(-2,2,21)
yrange=np.linspace(-2,2,21)

x,y=np.meshgrid(xrange,yrange)
z=x**2+y**2

ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

ax=plt.axes()
ax.contour(x,y,z)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()