import os
import numpy as np
import scipy.stats as ss

fn=os.path.join(os.path.dirname(__file__),"height.txt")
heights=np.loadtxt(fn)

mu=np.mean(heights)
sigma=np.std(heights)

p1=ss.norm.cdf(160,mu,sigma)
print("P(X<=160)=",p1)

p2=ss.norm.cdf(180,mu,sigma)
print("P(X>180)=",1-p2)