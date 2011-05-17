#ifndef MAT_TRI_H
#define MAT_TRI_H

#include <iostream>
#include "MatVecExcept.h"
#include "MatVecBase.h"
#include "MatVec.h"
#include "blas.h"

/************************************************************
 *  TRI Matrix specialization
 ************************************************************/
template<class Tdata>
class Matrix<Tdata,TRI> 
  : public Matrix_base<Matrix<Tdata,TRI> >{
private:
  typedef Matrix_base<Matrix<Tdata,TRI> > Mat_base;
  Tdata *arr_;
  size_t nrows_;
  size_t ncols_;
  bool mem_allocated;
  bool upper_; //true = data is stored in upper right triangle,
               //false= data is stored in lower left triangle
  bool trans_;
  bool unitdiag_;

  const static Tdata one = 1;
  const static Tdata zero = 0;
  
public:
  //requirements inherited from Matrix_base
  const Matrix& lhs() const{return *this;}

  const Matrix& rhs() const{return *this;}

  const Matrix& obj() const{return *this;}
  Matrix& obj(){return *this;}
  
  //constructors
  explicit Matrix(size_t nrows=0,size_t ncols=0,Tdata initval = 0);
  Matrix(size_t nrows,size_t ncols,Tdata *arr,
	 bool upper=true,bool trans=false,bool unitdiag=false);
  Matrix(const Matrix& M);
  
  template<class T>
  Matrix(const Matrix_base<T>& M) 
    : arr_(0),nrows_(0),ncols_(0),mem_allocated(0),
    upper_(1),trans_(0),unitdiag_(0){
    reallocate(M.nrows() ); *this = M.obj();}
  
  //destructor
  ~Matrix();
  
  //operator= : this needs to be explicitly defined, because
  //            otherwise the default will be used
  using Mat_base::operator=;
  Matrix& operator=(const Matrix& M){return Mat_base::operator=(M.obj());}
  
  //deep and shallow copy
  void deep_copy(const Matrix& M);
  void viewof(const Matrix& M);
  
  //memory (re)allocation
  void reallocate(size_t nrows,size_t ncols,Tdata initval = 0);
  
  //element access
  Tdata& operator()(size_t i, size_t j);
  const Tdata operator()(size_t i,size_t j) const;
  
  //row, column, and diagonal access
  Vector<Tdata,GEN> row(size_t i){
    throw NotImplementedError("MatrixSym::row()");}
  const Vector<Tdata,GEN> row(size_t i) const{
    throw NotImplementedError("MatrixSym::row()");}
  Vector<Tdata,GEN> col(size_t j){
    throw NotImplementedError("MatrixSym::col()");}
  const Vector<Tdata,GEN> col(size_t j) const{
    throw NotImplementedError("MatrixSym::col()");}
  Vector<Tdata,GEN> diag();
  const Vector<Tdata,GEN> diag() const;
  
  //access to private data
  Tdata* arr(){return arr_;}
  const Tdata* arr() const{return arr_;}
  size_t nrows() const {return trans_ ? ncols_ : nrows_;}
  size_t ncols() const {return trans_ ? nrows_ : ncols_;}
  size_t rank() const{return std::min(nrows_,ncols_);}
  size_t size() const{return nrows_*ncols_;}
  int inc() const{return 1;}
  size_t ld() const{return nrows_;}
  bool upper() const{return upper_;}
  bool unitdiag() const{return unitdiag_;}
  void flip_unitdiag(){unitdiag_ = !unitdiag_;}
  
  //Transpose
  Matrix Transpose() const{return Matrix(ncols_,nrows_,arr_,
					 upper_,!trans_,unitdiag_);}
  void TransposeSelf(){trans_ = !(trans_);}
  bool isTransposed() const{return trans_;}
  
  //misc routines
  void SetAllTo(Tdata val);
  Tdata SumElements() const;
  Tdata Trace(){return diag().SumElements();}
};

/************************************************************
  ostream definitions for symmetric matrix
************************************************************/

template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T,TRI> & M)
{
  os << M.nrows() << " x " << M.ncols();
  os << "(TRI)";
  if(M.isTransposed()) os<<"(Trans)";
  os << '\n';
  
  for(size_t i=0;i<M.nrows();i++){
    os << ' ';
    for(size_t j=0;j<M.ncols();j++)
      os << M(i,j) << ' ';
    os << '\n';
  }
  return os;
}

/************************************************************
  BLAS specializations for (double,TRI) matrices
************************************************************/
/*
typedef Matrix<double,TRI> DT_Mat;
typedef Matrix<double,DENSE> DD_Mat;
typedef Vector<double,GEN> DG_Vec;

//SCAL: matrix scaling : X *= a
template<>
inline DT_Mat& blas_SCAL(DS_Mat& M, double alpha){
  dscal( M.size(),alpha,M.arr(),M.inc() );
  return M;
}

//COPY: matrix copy : M1 = M2
template<>
inline DT_Mat& blas_COPY(DT_Mat& M1, const DT_Mat& M2){
  if(M1.nrows() != M2.nrows())
    throw MatVecError("COPY : matrix sizes must match");
  else if( M1.upper() != M2.upper() )
    throw NotImplementedError("COPY : upper to lower sym matrix");
  else if(M1.arr() != M2.arr())
    dcopy(M1.size(), M2.arr(), M2.inc(), M1.arr(), M1.inc());
  return M1;
}

//AXPY: matrix addition : y += a*x
template<>
inline DT_Mat& blas_AXPY(DT_Mat& Y, double alpha,  const DT_Mat& X){
  if( (Y.nrows() != X.nrows()) || (Y.ncols() != X.ncols()) )
    throw MatVecError("AXPY : matrix sizes must match");
  else if( Y.upper() != X.upper() )
    throw NotImplementedError("AXPY : upper + lower sym matrix");
  else
    daxpy(Y.size(), alpha, X.arr(), X.inc(), Y.arr(), Y.inc());
  return Y;
}


//MV: matrix-vector product : y = alpha*A*x + beta*y
template<>
inline DG_Vec& blas_MV(DG_Vec& y, double alpha, const DT_Mat& A, 
		       const DG_Vec& x, double beta){
  if((A.nrows() != y.size()) || (A.ncols() != x.size()) )
    throw MatVecError("MV : (sym)Matrix/Vector sizes must match");
  dsymv(A.upper(),A.nrows(),alpha, A.arr(), A.ld(),
	x.arr(), x.inc(), beta, y.arr(), y.inc());
  return y;
}

//MM: matrix-matrix product : C = alpha*A*B + beta*C
//      where A is symmetric
template<>
inline DD_Mat& blas_MM(DD_Mat& C, double alpha, const DT_Mat& A, 
		       const DD_Mat& B, double beta){
  if( (A.nrows() != C.nrows()) || (A.ncols()!=B.nrows()) 
      || (C.ncols() != B.ncols()) ){
    throw MatVecError("MM(sym) : Matrix sizes must match");
  }else{
    dsymm(true,A.upper(),C.nrows(),C.ncols(),alpha,A.arr(),
	  A.ld(),B.arr(),B.ld(),beta,C.arr(),C.ld());
  }
  return C;
}

//MM: matrix-matrix product : C = alpha*B*A + beta*C
//      where A is symmetric
template<>
inline DD_Mat& blas_MM(DD_Mat& C, double alpha, const DD_Mat& B, 
		       const DT_Mat& A, double beta){
  if( (B.nrows() != C.nrows()) || (B.ncols()!=A.nrows()) 
      || (C.ncols() != B.ncols()) ){
    throw MatVecError("MM(sym) : Matrix sizes must match");
  }else{
    dsymm(false,A.upper(),C.nrows(),C.ncols(),alpha,A.arr(),
	  A.ld(),B.arr(),B.ld(),beta,C.arr(),C.ld());
  }
  return C;
}

//MM: matrix-matrix product : C = alpha*A^T*A + beta*C 
//                                or    A*A^T
//      where C is symmetric
template<>
inline DT_Mat& blas_MM(DT_Mat& C, double alpha, const DD_Mat& A, 
		       const DD_Mat& B, double beta){
  if( A.arr() != B.arr() || A.isTransposed() == B.isTransposed() 
      || A.ncols()!=B.nrows() || A.nrows()!=B.ncols() )
    throw NotImplementedError("MM : C(sym) = AB : B must equal A^T");
  if( (A.nrows() != C.nrows()) || (C.ncols() != B.ncols()) ){
    throw MatVecError("MM(sym) : Matrix sizes must match");
  }else{
    dsyrk(C.upper(),A.isTransposed(),C.nrows(),A.ncols(),alpha,
	  A.arr(),A.ld(),beta,C.arr(),C.ld());
  }
  return C;
}
*/

#endif /*MAT_TRI_H*/
