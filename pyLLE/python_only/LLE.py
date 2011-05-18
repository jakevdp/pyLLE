import numpy
import pylab

######################################################################
#  Locally Linear Embedding
######################################################################

USE_SVD = True

def dimensionality(M,k,v=0.9,quiet=True):
    M = numpy.matrix(M)
    d,N = M.shape
    assert k<N
    m_estimate = []
    var_total = 0
    for row in range(N):
        if row%500==0:print 'finished %s out of %s' % (row,N)
        #-----------------------------------------------
        #  find k nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.array(M-M[:,row])
        vec = (M_Mi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #compute distances
        x = numpy.matrix(M[:,nbrs] - M[:,row])
        #singular values of x give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(x,compute_uv=0))**2

        #sig2 is sorted from large to small
        
        #use sig2 to compute intrinsic dimensionality of the
        # data at this neighborhood.  The dimensionality is the
        # number of eigenvalues needed to sum to the total
        # desired variance
        sig2 /= sig2.sum()
        S = sig2.cumsum()
        m = S.searchsorted(v)
        if m>0:
            m += ( (v-S[m-1])/sig2[m] )
        else:
            m = v/sig2[m]
        m_estimate.append(m)
        
        r = numpy.sum(sig2[m:])
        var_total += r

    if not quiet: print 'average variance conserved: %.3g' % (1.0 - var_total/N)

    return m_estimate


def LLE(M,k,m,quiet=False):
    """
    Perform a Locally Linear Embedding analysis on M
    
    >> LLE(M,k,d,quiet=False)
    
     - M is a numpy array of rank (d,N), consisting of N
        data points in d dimensions.

     - k is the number of neighbors to use in the embedding

     - m is the number of dimensions to which the dataset will
        be reduced.

    Based on the algorithm outlined in
     'An Introduction to Locally Linear Embedding'
        by L. Saul and S. Roewis

    Using imrovements suggested in
     'Locally Linear Embedding for Classification'
        by D. deRidder and R.P.W. Duin
    """
    M = numpy.matrix(M)
    d,N = M.shape
    assert k<N
    if not quiet:
        print 'performing LLE on %i points in %i dimensions...' % (N,d)

    #build the weight matrix
    W = numpy.zeros((N,N))

    if not quiet:
        print ' - constructing [%i x %i] weight matrix...' % W.shape

    m_estimate = []
    var_total = 0.0
    
    for row in range(N):
        #-----------------------------------------------
        #  find k nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.array(M-M[:,row])
        vec = (M_Mi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #-----------------------------------------------
        #  compute weight vector based on neighbors
        #-----------------------------------------------

        #compute covariance matrix of distances
        M_Mi = numpy.matrix(M_Mi[:,nbrs])
        Q = M_Mi.T * M_Mi

        #singular values of M_Mi give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(M_Mi,compute_uv=0))**2

        #use sig2 to compute intrinsic dimensionality of the
        # data at this neighborhood.  The dimensionality is the
        # number of eigenvalues needed to sum to the total
        # desired variance
        v=0.9
        sig2 /= sig2.sum()
        S = sig2.cumsum()
        m_est = S.searchsorted(v)
        if m_est>0:
            m_est += ( (v-S[m_est-1])/sig2[m_est] )
        else:
            m_est = v/sig2[m_est]
        m_estimate.append(m_est)
        
        #Covariance matrix may be nearly singular:
        # add a diagonal correction to prevent numerical errors
        # correction is equal to the sum of the (d-m) unused variances
        #  (as in deRidder & Duin)
        r = numpy.sum(sig2[m:])
        var_total += r
        Q.flat[::k+1] += r
        #Note that Roewis et al instead uses "a correction that 
        #   is small compared to the trace":
        #r = 0.001 * float(Q.trace())
    
        #solve for weight
        w = numpy.linalg.solve(Q,numpy.ones(Q.shape[0]))
        w /= numpy.sum(w)

        #update row of the weight matrix
        W[row,nbrs] = w

    if not quiet:
        print ' - finding [%i x %i] null space of weight matrix...' % (m,N)
    #to find the null space, we need the bottom d+1
    #  eigenvectors of (W-I).T*(W-I)
    #Compute this using the svd of (W-I):
    I = numpy.identity(W.shape[0])
    U,sig,VT = numpy.linalg.svd(W-I,full_matrices=0)
    indices = numpy.argsort(sig)[1:m+1]

    print 'm_estimate: %.2f +/- %.2f' % (numpy.median(m_estimate),numpy.std(m_estimate))
    print 'average variance conserved: %.3g' % (1.0 - var_total/N)
    
    return numpy.array(VT[indices,:])





######################################################################
#  Modified Locally Linear Embedding
######################################################################

def MLLE(X,k,d_out,TOL = 1E-12):
    """
    perfrom Modified LLE on X
    """
    
    X = numpy.asarray(X)
    d_in,N = X.shape
    assert d_out < d_in
    assert k >= d_out
    assert k < N

    #some variables to hold needed values
    rho = numpy.zeros(N)
    w_reg = numpy.zeros([N,k])
    evals = numpy.zeros([N,k])
    V = [0 for i in range(N)]
    neighbors = numpy.zeros([N,k],dtype=int)

    #some functions to simplify the code
    column_vector = lambda x: x.reshape( (x.size,1) )
    one = lambda d: numpy.ones((d,1))

    for i in range(N):
        #find neighbors
        X_Xi = X - column_vector( X[:,i] )
        neighbors[i] = numpy.argsort( (X_Xi**2).sum(0) )[1:k+1]

        #find regularized weights: this is like normal LLE
        Gi = X_Xi[ : , neighbors[i] ]
        Qi = numpy.dot(Gi.T,Gi)

        Qi.flat[::k+1] += 1E-3 #* Qi.trace()

        y = numpy.linalg.solve(Qi,numpy.ones(k))
        w_reg[i] = y/y.sum()

        #find the eigenvectors and eigenvalues of Gi.T*Gi
        # using SVD
        # we want V[i] to be a [k x k] matrix, where the columns
        # are eigenvectors of Gi^T * G
        V[i],sig,UT = numpy.linalg.svd(Gi.T)
        evals[i][:len(sig)] = sig**2

        #compute rho_i : this is used to determine eta, the
        # cutoff used to determine the size of the "almost null"
        # space of the local covariance matrices.
        rho[i] = (evals[i,d_out:]).sum() / (evals[i,:d_out]).sum()

    #find eta - the median of the N rho values
    rho.sort()
    eta = rho[int(N/2)]

    #The next loop calculates Phi.
    # This is the [N x N] matrix whose null space is the desired embedding
    Phi = numpy.zeros( (N,N) )
    for i in range(N):
        #determine si - the size of the largest set of eigenvalues
        # of Qi such that satisfies:
        #    sum(in_set)/sum(not_in_set) < eta
        # with the constraint that 0<si<=k-d_out

        si = 1
        while si < k-d_out:
            this_eta = sum( evals[i,k-si:] ) / sum( evals[i,:k-si] )
            if this_eta > eta:
                if(si!=1): si -= 1
                break
            else:
                si+=1

        #select bottom si eigenvectors of Qi
        # and calculate alpha
        Vi = V[i][:,k-si:]
        alpha_i = numpy.linalg.norm( Vi.sum(0) )/numpy.sqrt(si)

        #compute Householder matrix which satisfies
        #  Hi*Vi.T*one(k) = alpha_i*one(s)
        # using proscription from paper
        h = alpha_i * one(si) - numpy.dot(Vi.T,one(k))

        nh = numpy.linalg.norm(h)
        if nh < TOL:
            h = numpy.zeros( (si,1) )
        else:
            h /= nh
            
        Hi = numpy.identity(si) - 2*numpy.dot(h,h.T)

        Wi = numpy.dot(Vi,Hi) + (1-alpha_i) * column_vector(w_reg[i])

        W_hat = numpy.zeros( (N,si) )
        W_hat[neighbors[i],:] = Wi
        W_hat[i]-=1
            
        Phi += numpy.dot(W_hat,W_hat.T)
        
    U,sig,VT = numpy.linalg.svd(Phi)
    return VT[-d_out-1:-1]

def new_LLE_pts(M,M_LLE,k,x):
    """
    inputs:
       - M: a rank [d * N] data-matrix
       - M_LLE: a rank [m * N] matrixwhich is the output of LLE(M,k,m)
       - k: the number of neighbors used to produce M_LLE
       - x: a length d data vector OR a rank [d * Nx] array
    returns:
       - y: the LLE reconstruction of x
    """
    M = numpy.matrix(M)
    M_LLE = numpy.matrix(M_LLE)

    d,N = M.shape
    m,N2 = M_LLE.shape
    assert N==N2

    #make sure x is a column vector
    if numpy.rank(x) == 1:
        x = numpy.matrix(x).T
    else:
        x = numpy.matrix(x)
    assert x.shape[0] == d
    Nx = x.shape[1]

    W = numpy.matrix(numpy.zeros([Nx,N]))

    for i in range(x.shape[1]):
        #  find k nearest neighbors
        M_xi = numpy.array(M-x[:,i])
        vec = (M_xi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #compute covariance matrix of distances
        M_xi = numpy.matrix(M_xi[:,nbrs])
        Q = M_xi.T * M_xi

        #singular values of x give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(M_xi,compute_uv=0))**2
    
        #Covariance matrix may be nearly singular:
        # add a diagonal correction to prevent numerical errors
        # correction is equal to the sum of the (d-m) unused variances
        #  (as in deRidder & Duin)
        r = numpy.sum(sig2[m:])
        Q += r*numpy.identity(Q.shape[0])
        #Note that Roewis et al instead uses "a correction that 
        #   is small compared to the trace":
        #r = 0.001 * float(Q.trace())
    
        #solve for weight
        w = numpy.linalg.solve(Q,numpy.ones((Q.shape[0],1)))[:,0]
        w /= numpy.sum(w)

        W[i,nbrs] = w
        print 'x[%i]: variance conserved: %.2f' % (i,1.0- sig2[m:].sum())

    #multiply weights by projections of neighbors to get y
    print M_LLE.shape
    print W.shape
    print len(nbrs)
    
    return numpy.array( M_LLE  * numpy.matrix(W).T )





######################################################################
#  Hessian Locally Linear Embedding
######################################################################

def HLLE(M,k,d,quiet=False):
    """
    Perform a Hessian Eigenmapping analysis on M

    >> HLLE(M,k,d,quiet=False)
    
     - M is a numpy array of rank (dim,N), consisting of N
        data points in dim dimensions.

     - k is the number of neighbors to use in the embedding

     - d is the number of dimensions to which the dataset will
        be reduced.
    
    Implementation based on algorithm outlined in
     'Hessian Eigenmaps: new locally linear embedding techniques
      for high-dimensional data'
        by C. Grimes and D. Donoho, March 2003
    """
    M = numpy.matrix(M)
    dim,N = M.shape
    
    if not quiet:
        print 'performing HLLE on %i points in %i dimensions...' % (N,dim)
    
    dp = d*(d+1)/2
    W = numpy.matrix( numpy.zeros([dp*N,N]) )
    
    if not quiet:
        print ' - constructing [%i x %i] weight matrix...' % W.shape
        
    for i in range(N):
        #-----------------------------------------------
        #  find nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.array(M-M[:,i])
        vec = sum(M_Mi*M_Mi,0)
        nbrs = numpy.argsort(vec)[1:k+1]

        #-----------------------------------------------
        #  center the neighborhood using the mean
        #-----------------------------------------------
        nbrhd = M[:,nbrs]
        nbrhd -= nbrhd.mean(1)

        #-----------------------------------------------
        #  compute local coordinates
        #   using a singular value decomposition
        #-----------------------------------------------
        U,vals,VT = numpy.linalg.svd(nbrhd,full_matrices=0)
        nbrhd = numpy.matrix( (VT.T)[:,:d] )

        #-----------------------------------------------
        #  build Hessian estimator
        #-----------------------------------------------
        ct = 0
        Yi = numpy.matrix(numpy.zeros([k,dp]))
        
        for mm in range(d):
            for nn in range(mm,d):
                Yi[:,ct] = numpy.multiply(nbrhd[:,mm],nbrhd[:,nn])
                ct += 1
        Yi = numpy.concatenate( [numpy.tile(1,(k,1)), nbrhd, Yi],1 )

        #-----------------------------------------------
        #  orthogonalize linear and quadratic forms
        #   with QR factorization
        #  make the weights sum to 1
        #-----------------------------------------------
        Q,R = mgs(Yi)
        w = numpy.array(Q[:,d+1:].T)
        S = w.sum(1) #sum along rows

        #if S[i] is too small, set it equal to 1.0
        S[numpy.where(numpy.abs(S)<0.0001)] = 1.0
        W[ i*dp:(i+1)*dp , nbrs ] = (w.T/S).T

    #-----------------------------------------------
    # To find the null space, we want the
    #  first d+1 eigenvectors of W.T*W
    # Compute this using an svd of W
    #-----------------------------------------------
    if not quiet:
        print ' - computing [%i x %i] null space of weight matrix...' % (d,N)

    #Fast, but memory intensive
    if USE_SVD:
        U,sig,VT = numpy.linalg.svd(W,full_matrices=0)
        del U
        indices = numpy.argsort(sig)[1:d+1]
        Y = VT[indices,:] * numpy.sqrt(N)

    #Slower, but uses less memory
    else:
        C = W.T*W
        del W
        sig2,V = numpy.linalg.eigh(C)
        del C
        indices = range(1,d+1) #sig2 is sorted in ascending order
        Y = V[:,indices].T * numpy.sqrt(N)

    #-----------------------------------------------
    # Normalize Y
    #  we need R = (Y.T*Y)^(-1/2)
    #   do this with an SVD of Y
    #      Y = U*sig*V.T
    #      Y.T*Y = (V*sig.T*U.T) * (U*sig*V.T)
    #            = U*(sig*sig.T)*U.T
    #   so
    #      R = V * sig^-1 * V.T
    #-----------------------------------------------
    if not quiet:
        print ' - normalizing null space via SVD...'

    #Fast, but memory intensive
    if USE_SVD:
        U,sig,VT = numpy.linalg.svd(Y,full_matrices=0)
        del U
        S = numpy.matrix(numpy.diag(sig**-1))
        R = VT.T * S * VT
        return numpy.array(Y*R)

    #Slower, but uses less memory
    else:
        C = Y*Y.T
        sig2,U = numpy.linalg.eigh(C)
        U = U[:,::-1] #eigenvectors should be in descending order
        sig2=sig2[::-1]
        S = numpy.matrix(numpy.zeros(U.shape))
        for i in range(d):
            S[i,i] = (1.0*sig2[i])**-1.5
        return numpy.array(C * U * S * U.T * Y)




######################################################################
#  Modified Gram-Schmidt
######################################################################

def mgs(A):
    """
    Modified Gram-Schmidt version of QR factorization

    returns matrices Q,R such that A = Q*R
    where Q is an orthogonal matrix,
          R is an upper-right triangular matrix
    """
    #copy A and make sure it's a matrix
    Q = 1.0*numpy.matrix(A)
    m,n = Q.shape
    #assume m>=n
    R = numpy.matrix(numpy.zeros([n,n]))
    for i in range(n):
        v = Q[:,i]
        R[i,i] = numpy.sqrt(numpy.sum(numpy.multiply(v,v)))
        Q[:,i] /= R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,i].T * Q[:,j]
            Q[:,j] -= R[i,j] * Q[:,i]

    return Q,R
