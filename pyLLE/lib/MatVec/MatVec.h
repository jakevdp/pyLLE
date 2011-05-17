#ifndef MATVEC_H
#define MATVEC_H

/************************************************************
  Matrix/Vector template class

 written by Jake VanderPlas, January 2009

  --------

    Note: using column-major format to be compatible
          with LAPACK and BLAS
************************************************************/

#include <iostream>
#include "MatVecExcept.h"
#include "MatVecBase.h"
#include "blas.h"

/************************************************************
 *  DENSE Matrix specialization
 ************************************************************/
template<class Tdata>
class Matrix<Tdata,DENSE> 
  : public Matrix_base<Matrix<Tdata,DENSE> >{
private:
  typedef Matrix_base<Matrix<Tdata,DENSE> > Mat_base;
  Tdata *arr_;
  size_t nrows_;
  size_t ncols_;
  bool mem_allocated;
  bool trans_;
  
public:
  //requirements inherited from Matrix_base
  const Matrix& lhs() const{return *this;}

  const Matrix& rhs() const{return *this;}

  const Matrix& obj() const{return *this;}
  Matrix& obj(){return *this;}
  
  //constructors
  explicit Matrix(size_t nrows=0,size_t ncols=0,Tdata initval = 0);
  Matrix(size_t nrows,size_t ncols,Tdata *arr,bool trans=false);
  Matrix(const Matrix& M);
  
  template<class T>
  Matrix(const Matrix_base<T>& M) 
    : arr_(0), nrows_(0), ncols_(0), mem_allocated(0), trans_(0){
    reallocate(M.nrows(),M.ncols() ); *this = M.obj();}
  
  //destructor
  ~Matrix();
  
  //operator= : this needs to be explicitly defined, because
  //            otherwise the default will be used
  using Mat_base::operator=;
  Matrix& operator=(const Matrix& M){return Mat_base::operator=(M.obj());}
  
  //deep and shallow copy
  void deep_copy(const Matrix& M);
  void viewof(const Matrix& M);
  void swap(Matrix& M);
  
  //memory (re)allocation
  void reallocate(size_t nrows,size_t ncols,Tdata initval = 0);
  void set_mem_allocated(bool TF){mem_allocated = TF;}
  
  //element access
  Tdata& operator()(size_t i, size_t j);
  const Tdata& operator()(size_t i,size_t j) const;

  const Tdata& get(size_t i, size_t j) const{ return this->operator()(i,j); }
  void set(size_t i, size_t j, Tdata val){this->operator()(i,j)=val;}
  
  //row, column, and diagonal access
  Vector<Tdata,GEN> row(size_t i);
  const Vector<Tdata,GEN> row(size_t i) const;
  Vector<Tdata,GEN> col(size_t j);
  const Vector<Tdata,GEN> col(size_t j) const;
  Vector<Tdata,GEN> diag();
  const Vector<Tdata,GEN> diag() const;
  
  //access to private data
  Tdata* arr(){return arr_;}
  const Tdata* arr() const{return arr_;}
  size_t nrows() const {return trans_ ? ncols_ : nrows_;}
  size_t ncols() const {return trans_ ? nrows_ : ncols_;}
  size_t rank() const{return (nrows_<ncols_) ? nrows_ : ncols_;}
  size_t size() const{return nrows_*ncols_;}
  int inc() const{return 1;}
  size_t ld() const{return nrows_;}
  
  //Transpose
  Matrix Transpose() const{return Matrix(ncols(),nrows(),arr_,!trans_);}
  void TransposeSelf(){trans_ = !(trans_);}
  bool isTransposed() const{return trans_;}
  
  //misc routines
  void SetAllTo(Tdata val);
  Tdata SumElements() const;
  Tdata Trace(){return diag().SumElements();}
};

/************************************************************
 *  GEN Vector specialization
 ************************************************************/
template<class Tdata>
class Vector<Tdata,GEN> 
  : public Vector_base<Vector<Tdata,GEN> >{
private:
  typedef Vector_base<Vector<Tdata,GEN> > Vec_base;
  Tdata *arr_;
  int inc_;
  size_t size_;
  bool mem_allocated;
    
public:
  //requirements inherited from Vector_base
  const Vector& lhs() const{return *this;}

  const Vector& rhs() const{return *this;}

  const Vector& obj() const{return *this;}
  Vector& obj(){return *this;}
  
  //constructors
  explicit Vector(size_t size=0, Tdata initval=0);
  Vector(size_t size,Tdata *arr,int inc=1);
  Vector(const Vector& V);
  
  template<class T>
  Vector(const Vector_base<T>& V) 
    : arr_(0), inc_(0), size_(0), mem_allocated(0){
    reallocate(V.size()); *this = V.obj();}
  
  //destructor
  ~Vector();
  
  //operator= : this needs to be explicitly defined, because
  //            otherwise the default will be used
  using Vec_base::operator=;
  Vector& operator=(const Vector& V){return Vec_base::operator=(V);}
  
  //deep and shallow copy
  void deep_copy(const Vector& V);
  void viewof(const Vector& V);

  //swap elements
  void swap(size_t,size_t);
  
  //memory (re)allocation
  void reallocate(size_t size,Tdata initval=0);
  void set_mem_allocated(bool TF){mem_allocated = TF;}
  
  //element access
  Tdata& operator()(size_t i);
  const Tdata& operator()(size_t i) const;

  const Tdata& get(size_t i) const{ return this->operator()(i); }
  void set(size_t i, Tdata val){this->operator()(i)=val;}
  
  Vector SubVector(size_t,size_t);
  
  //access to private data
  Tdata *arr(){return arr_;}
  const Tdata *arr() const{return arr_;}
  size_t size() const{return size_;}
  int inc() const{return inc_;}
  
  //misc routines
  void SetAllTo(Tdata val);
  Tdata SumElements() const;
};

/************************************************************
  ostream definitions for matrix & vector
************************************************************/

template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T,DENSE> & M)
{
  os << M.nrows() << " x " << M.ncols();
  if(M.isTransposed() )
    os << "(T)";
  os << '\n';
  
  for(size_t i=0;i<M.nrows();i++){
    os << ' ';
    for(size_t j=0;j<M.ncols();j++)
      os << M(i,j) << ' ';
    os << '\n';
  }
  return os;
}

template<class T>
std::ostream& operator<<(std::ostream& os, const Vector<T,GEN> & V)
{
  os << V.size() << '\n';
  for(size_t i=0;i<V.size();i++)
    os << V(i) << ' ';
  os << '\n';
  return os;
}

/************************************************************
  BLAS specializations for (int,DENSE) matrices
                       and (int,GEN)   vectors
************************************************************/
//typedef Vector<int,GEN> IG_Vec;
//typedef Matrix<int,DENSE> ID_Mat;

//COPY: vector/matrix copy : V1 = V2
template<>
inline Vector<int,GEN>& blas_COPY(Vector<int,GEN>& V1, 
				  const Vector<int,GEN>& V2){
  if( V1.size() != V2.size() )
    throw MatVecError("COPY : vector sizes must match");
  if(V1.arr() != V2.arr())
    for(size_t i=0;i<V1.size();i++)
      V1(i) = V2(i);
  return V1;
}

template<>
inline Matrix<int,DENSE>& blas_COPY(Matrix<int,DENSE>& M1, 
				    const Matrix<int,DENSE>& M2){
  if( (M1.nrows() != M2.nrows()) || (M1.ncols() != M2.ncols()) )
    throw MatVecError("COPY : matrix sizes must match");
  if(M1.arr() != M2.arr()){
    for(size_t i=0;i<M1.nrows();i++)
      for(size_t j=0;j<M1.ncols();j++)
	M1(i,j) = M2(i,j);
  }
  return M1;
}

/************************************************************
  BLAS specializations for (double,DENSE) matrices
                       and (double,GEN)   vectors
************************************************************/
//typedef Vector<double,GEN> DG_Vec;
//typedef Matrix<double,DENSE> DD_Mat;

//DOT: vector dot product
template<>
inline double blas_DOT(const Vector<double,GEN>& V1, 
		       const Vector<double,GEN>& V2){
  if( V1.size() != V2.size() )
    throw MatVecError("DOT : vector sizes must match");
  return ddot(V1.size(),V1.arr(),V1.inc(),V2.arr(),V2.inc());
}


//NRM2: vector/matrix 2-norm : sqrt( sum(elements^2) )
template<>
inline double blas_NRM2(const Matrix<double,DENSE>& M){
  return dnrm2(M.size(),M.arr(),M.inc());
}

template<>
inline double blas_NRM2(const Vector<double,GEN>& V){
  return dnrm2(V.size(),V.arr(),V.inc());
}


//SCAL: vector/matrix scaling : X *= a
template<>
inline Matrix<double,DENSE>& blas_SCAL(Matrix<double,DENSE>& M, double alpha){
  dscal( M.size(),alpha,M.arr(),M.inc() );
  return M;
}

template<>
inline Vector<double,GEN>& blas_SCAL(Vector<double,GEN>& V,double alpha){
  dscal(V.size(),alpha,V.arr(),V.inc());
  return V;
}

//COPY: vector/matrix copy : V1 = V2
template<>
inline Vector<double,GEN>& blas_COPY(Vector<double,GEN>& V1, 
				     const Vector<double,GEN>& V2){
  if( V1.size() != V2.size() )
    throw MatVecError("COPY : vector sizes must match");
  if(V1.arr() != V2.arr())
    dcopy(V1.size(), V2.arr(), V2.inc(), V1.arr(), V1.inc() );
  return V1;
}

template<>
inline Matrix<double,DENSE>& blas_COPY(Matrix<double,DENSE>& M1, 
				       const Matrix<double,DENSE>& M2){
  if( (M1.nrows() != M2.nrows()) || (M1.ncols() != M2.ncols()) )
    throw MatVecError("COPY : matrix sizes must match");
  if( M1.isTransposed() != M2.isTransposed() ){
    if(M1.ncols() > M1.nrows()){
      for(size_t i=0;i<M1.nrows();i++){
	M1.row(i) = M2.row(i);
      }
    }else{
      for(size_t i=0;i<M1.ncols();i++){
	M1.col(i) = M2.col(i);
      }
    }
  }else if(M1.arr() != M2.arr()){
    dcopy(M1.size(), M2.arr(), M2.inc(), M1.arr(), M1.inc());
  }
  return M1;
}

//AXPY: matrix/vector addition : y += a*x
template<>
inline Vector<double,GEN>& blas_AXPY(Vector<double,GEN>& Y,double alpha,
				     const Vector<double,GEN>& X){
  if(X.size() != Y.size())
    throw MatVecError("AXPY : vector sizes must match");
  daxpy(Y.size(), alpha, X.arr(), X.inc(), Y.arr(), Y.inc());
  return Y;
}

template<>
inline Vector<double,GEN>& blas_AXPY(Vector<double,GEN>& Y,double alpha,
				     const double& X){
  daxpy(Y.size(), alpha, &X, 0, Y.arr(), Y.inc());
  return Y;
}

template<>
inline Matrix<double,DENSE>& blas_AXPY(Matrix<double,DENSE>& Y, double alpha,  
				       const Matrix<double,DENSE>& X){
  if( (Y.nrows() != X.nrows()) || (Y.ncols() != X.ncols()) )
    throw MatVecError("AXPY : matrix sizes must match");
  if( Y.isTransposed() != X.isTransposed() ){
    if(Y.ncols() > Y.nrows()){
      for(size_t i=0;i<Y.nrows();i++){
	Y.row(i) = alpha*X.row(i);
      }
    }else{
      for(size_t i=0;i<Y.ncols();i++){
	Y.col(i) = alpha*X.col(i);
      }
    }
  }else{
    daxpy(Y.size(), alpha, X.arr(), X.inc(), Y.arr(), Y.inc());
  }
  return Y;
}

template<>
inline Matrix<double,DENSE>& blas_AXPY(Matrix<double,DENSE>& Y,double alpha,
				       const double& X){
  daxpy(Y.size(), alpha, &X, 0, Y.arr(), Y.inc());
  return Y;
}


//MV: matrix-vector product : y = alpha*A*x + beta*y
template<>
inline Vector<double,GEN>& blas_MV(Vector<double,GEN>& y, double alpha, 
				   const Matrix<double,DENSE>& A, 
		       const Vector<double,GEN>& x, double beta){
  if((A.nrows() != y.size()) || (A.ncols() != x.size()) )
    throw MatVecError("MV : Matrix/Vector sizes must match");
  dgemv(A.isTransposed(),A.nrows(), A.ncols(), 
	alpha, A.arr(), A.ld(),
	x.arr(), x.inc(), beta, y.arr(), y.inc());
  return y;
}

//MM: matrix-matrix product : C = alpha*A*B + beta*C
template<>
inline Matrix<double,DENSE>& blas_MM(Matrix<double,DENSE>& C, double alpha, 
				     const Matrix<double,DENSE>& A, 
		const Matrix<double,DENSE>& B, double beta){
  if( (A.nrows() != C.nrows()) || (A.ncols()!=B.nrows()) 
      || (C.ncols() != B.ncols()) ){
    throw MatVecError("MM : Matrix sizes must match");
  }
  if(C.isTransposed()){
    //C^T = AB -> C = B^T A^T
    dgemm(!(B.isTransposed() ),!(A.isTransposed() ),
	  C.ncols(),C.nrows(),B.nrows(),alpha,B.arr(),B.ld(),
	  A.arr(),A.ld(),beta,C.arr(),C.ld());
  }else{
    dgemm(A.isTransposed(),B.isTransposed(),
	  C.nrows(),C.ncols(),A.ncols(),alpha,A.arr(),A.ld(),
	  B.arr(),B.ld(),beta,C.arr(),C.ld());
  }
  return C;
}

#endif /* MATVEC_H */
