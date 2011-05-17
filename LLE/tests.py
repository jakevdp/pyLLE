import pylab
import numpy

import os
import sys
sys.path.append(os.path.abspath('wrappers'))

import LLE
from datasets import Scurve

def test_neighbors(N,k,Nshow):
    X_3, theta = Scurve(N)

    neighbors = LLE.compute_neighbors(X_3,k,True,1)
    print neighbors[:,:Nshow]

    neighbors_2 = numpy.zeros((k,Nshow),dtype=int)
    for i in range(Nshow):
        d = ((X_3.T-X_3[:,i])**2).sum(1)
        neighbors_2[:,i] = numpy.argsort(d)[1:k+1]
    print neighbors_2

def test_LLE(N,k):
    numpy.random.seed(0)
    X_3, theta = Scurve(N)

    for func in (LLE.LLE,
                 LLE.HLLE,
                 LLE.MLLE):
        X_2 = func(X_3,k,2,verbose=3)
    
        pylab.figure()
        pylab.scatter(X_2[0], X_2[1], c=theta)

    pylab.show()

if __name__ == '__main__':
    #test_neighbors(1000,5,4)
    test_LLE(500,15)

