""" Cython bindings for the C++ LLE code.
"""
# Author: Jake Vanderplas
# License: BSD

import numpy as np
cimport numpy as np  
from cython.operator cimport dereference as deref
from libcpp cimport bool

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int64
ctypedef int ITYPE_t

######################################################################
# MatVec wrappers
#
cdef extern from "MatVec.h":
    cdef cppclass Matrix[Tdata]:
        Matrix(size_t,size_t,Tdata)
        Matrix(size_t,size_t,Tdata*,bool)
        
        void SetAllTo(Tdata)
        Tdata SumElements()
        Tdata Trace()
        
        Tdata get(size_t,size_t)
        void set(size_t,size_t,Tdata)
        
        size_t nrows()
        size_t ncols()

    cdef cppclass Vector[Tdata]:
        Vector(size_t,Tdata)
        Vector(size_t,Tdata*,size_t)
        
        void SetAllTo(Tdata)
        Tdata SumElements()
        
        Tdata get(size_t)
        void set(size_t,Tdata)

        size_t size()

    cdef cppclass Mat_data_t "Matrix<double,DENSE>"
    cdef cppclass Vec_data_t "Vector<double,GEN>"
    cdef cppclass Mat_index_t "Matrix<int,DENSE>"
    cdef cppclass Vec_index_t "Vector<int,GEN>"



######################################################################
# Routines for converting between ndarray and Vector/Matrix objects
#  
#  ndarray_from_mat
#  ndarray_from_vec
#
#  mat_from_ndarray
#  vec_from_ndarray
#
# Beware!! for [mat/vec]_from_ndarray, with view=True, these don't 
#  correctly count references. If you were to create a Matrix or Vector
#  object from an ndarray, then delete the original ndarray, 
#  the Matrix object would be pointing to deallocated memory. 
#
# Note that there is no view mode for ndarray_from_[mat/vec].  The
#  thing to do instead is to allocate an ndarray, create a Matrix or
#  Vector view, and then modify this.
#
# This is all a bit clumsy.  I'd like to do away with MatVec completely, 
#  and rewrite this code using simple c++ wrappers of ndarray containers.
#  Some day...
#
cdef np.ndarray[DTYPE_t,ndim=2] ndarray_from_mat(Matrix[DTYPE_t] *m):
    cdef size_t i, j
    m_n = np.empty( (m.nrows(),m.ncols()), dtype=DTYPE )
    for i in range(m_n.shape[0]):
        for j in range(m_n.shape[1]):
            m_n[i,j] = m.get(i,j)
    return m_n

cdef Matrix[DTYPE_t] *mat_from_ndarray_d(np.ndarray[DTYPE_t,ndim=2] m,
                                         bool view = 1):
    cdef size_t i, j
    cdef Matrix[DTYPE_t] *mret 
    cdef bool transpose 
    cdef DTYPE_t* buf = <DTYPE_t*>m.data
    
    if view:
        m = np.ascontiguousarray(m)
        transpose = m.flags['C_CONTIGUOUS']
        mret = new Matrix[DTYPE_t](m.shape[0],m.shape[1],
                                   buf,transpose)
    else:
        mret = new Matrix[DTYPE_t](m.shape[0],m.shape[1],0.0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                mret.set(i,j,m[i,j])
    return mret

cdef Matrix[ITYPE_t] *mat_from_ndarray_i(np.ndarray[ITYPE_t,ndim=2] m,
                                         bool view = 1):
    cdef size_t i, j
    cdef Matrix[ITYPE_t] *mret 
    cdef bool transpose 
    cdef ITYPE_t* buf = <ITYPE_t*>m.data
    
    if view:
        m = np.ascontiguousarray(m)
        transpose = m.flags['C_CONTIGUOUS']
        mret = new Matrix[ITYPE_t](m.shape[0],m.shape[1],
                                   buf,transpose)
    else:
        mret = new Matrix[ITYPE_t](m.shape[0],m.shape[1],0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                mret.set(i,j,m[i,j])
    return mret

cdef np.ndarray[DTYPE_t,ndim=1] ndarray_from_vec(Vector[DTYPE_t] *v):
    cdef size_t i
    v_n = np.empty( v.size(), dtype=DTYPE )
    for i in range(v_n.shape[0]):
        v_n[i] = v.get(i)
    return v_n

cdef Vector[DTYPE_t] *vec_from_ndarray(np.ndarray[DTYPE_t,ndim=1] v,
                                       bool view = True):
    cdef size_t i
    cdef Vector[DTYPE_t] *vret 
    cdef int inc=0
    if view:
        vret = new Vector[DTYPE_t](v.shape[0],
                                   <DTYPE_t*>v.data, 
                                   v.strides[0]/v.itemsize)
    else:
        vret= new Vector[DTYPE_t](v.shape[0],0)
        for i in range(v.shape[0]):
            vret.set(i,v[i])
    return vret

######################################################################
# LLE wrappers
#
cdef extern from "LLE.h":
    cdef void cLLE "LLE"(Matrix[double], int,
                         Matrix[double], int)
    
    cdef void cMLLE "MLLE"(Matrix[double], int,
                           Matrix[double], double,
                           int)
    
    cdef void cHLLE "HLLE"(Matrix[double], int,
                           Matrix[double], int)

    cdef void ccompute_neighbors "compute_neighbors"(Matrix[double],
                                                     Matrix[int],
                                                     bool,
                                                     int)

						     

def LLE(np.ndarray[DTYPE_t,ndim=2] training_data not None,
        int k,
        int d_out,
        int verbose=0):
    cdef int d_in = training_data.shape[0]
    cdef int N = training_data.shape[1]
    projection = np.empty( (d_out,N), dtype=DTYPE )
    
    assert k<N 
    assert d_out<d_in

    cLLE( deref( mat_from_ndarray_d(training_data) ),
          k,
          deref( mat_from_ndarray_d(projection) ),
          verbose )
    
    return projection

def MLLE(np.ndarray[DTYPE_t,ndim=2] training_data not None,
         int k,
         int d_out,
         double TOL=1E-12,
         int verbose=0):
    cdef int d_in = training_data.shape[0]
    cdef int N = training_data.shape[1]
    projection = np.empty( (d_out,N), dtype=DTYPE )
    
    assert k<N 
    assert d_out<d_in

    cMLLE( deref( mat_from_ndarray_d(training_data) ),
           k,
           deref( mat_from_ndarray_d(projection) ),
           TOL,
           verbose )
    
    return projection					     

def HLLE(np.ndarray[DTYPE_t,ndim=2] training_data not None,
         int k,
         int d_out,		       
         int verbose=0):
    cdef int d_in = training_data.shape[0]
    cdef int N = training_data.shape[1]
    projection = np.empty( (d_out,N), dtype=DTYPE )
    
    assert k<N 
    assert d_out<d_in

    cHLLE( deref( mat_from_ndarray_d(training_data) ),
          k,
          deref( mat_from_ndarray_d(projection) ),
          verbose )
    
    return projection

def compute_neighbors(np.ndarray[DTYPE_t,ndim=2] training_data not None,
                      int k,
                      bool use_tree=False,
                      int verbose=0):
    cdef int N = training_data.shape[1]
    assert k<N
    
    neighbors = np.empty( (k,N), dtype=np.int )

    ccompute_neighbors( deref( mat_from_ndarray_d(training_data) ),
                        deref( mat_from_ndarray_i(neighbors) ),
                        use_tree,
                        verbose )

    return neighbors
    


######################################################################
# test routines
#
def test_matvec():
    m_n = np.random.random((4,4))

    v_n = np.random.random(4)
    
    cdef Matrix[DTYPE_t] *m = mat_from_ndarray_d(m_n)
    cdef Vector[DTYPE_t] *v = vec_from_ndarray(v_n)

    print m.Trace()
    print v.SumElements()

    del m
    del v

    print m_n
    print v_n
