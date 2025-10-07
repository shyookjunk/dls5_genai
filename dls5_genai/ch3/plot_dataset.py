import os
import numpy as np
import matplotlib.pyplot as plt

fn=os.path.join(os.path.dirname(__file__),"height_weight.txt")
xs=np.loadtxt(fn)

print(xs.shape)

small_xs=xs[:500]

plt.scatter(small_xs[:,0],small_xs[:,1])
plt.xlabel("Height(cm)")
plt.ylabel("Weight(kg)")
plt.show()

mu=np.mean(xs,axis=0)
cov=np.cov(xs,rowvar=False)
