import os
import numpy as np
import matplotlib.pyplot as plt

fn=os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(fn)

print(xs.shape)
print(xs[0])

plt.scatter(xs[:,0], xs[:,1])
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.show()