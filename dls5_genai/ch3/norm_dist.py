import numpy as np
import numpy.linalg as nl

def normal(x,mu=0,sigma=1):
    y=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
    return y

def multivariate_normal(x,mu,sigma):
    det_sig=nl.det(sigma)
    inv_sig=nl.inv(sigma)
    D=len(x)
    z=1/np.sqrt((2*np.pi)**D*det_sig)
    y=z*np.exp(-(x-mu).T@inv_sig@(x-mu)/2.0)

    return y