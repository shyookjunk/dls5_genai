import numpy as np
import matplotlib.pyplot as plt
import norm_dist as nd

x=np.linspace(-5,5,100)
y=nd.normal(x)
plt.plot(x,y)
plt.show()
