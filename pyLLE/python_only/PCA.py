#!/net/python/bin/python

import numpy

def PCA(M,n=None,frac=None):
    """
    given a rank (d,N) matrix M
    of N data points in d dimensions,
    perform a PCA dimensionality reduction of M
     n gives the minimum number of vectors to keep
     frac gives the minimum amount of variance to keep
    """
    M = numpy.asarray(M)
    
    if n==None:
        n = M.shape[0]

    if frac==None:
        frac = 1.0

    assert frac <= 1.0
    assert frac >= 0.0
    assert n <= M.shape[0]

    U,sig,V = numpy.linalg.svd(M,full_matrices = 0)

    sig_frac = sig**2/(sig**2).sum()

    #for val in sig_frac:
    #    print '%.6f'%val

    return V
