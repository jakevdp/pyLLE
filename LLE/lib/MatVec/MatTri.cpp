#include "MatTri.h"


/************************************************************
 * Matrix Definitions
 ************************************************************/

template<class T>
Matrix<T,TRI>::Matrix(size_t nrows/*=0*/,size_t ncols/*=0*/,T initval/*=0*/)
  : nrows_(nrows),ncols_(ncols),arr_(0),
    mem_allocated(false), upper_(true), trans_(false), unitdiag_(false){
  if(nrows_*ncols_ == 0){
    nrows_ = 0;
    ncols_ = 0;
  }else{
    arr_ = new T[nrows_ * ncols_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}


template<class T>
Matrix<T,TRI>::Matrix(size_t nrows,size_t ncols,T *arr,
		      bool upper,bool trans,bool unitdiag)
  : arr_(arr),nrows_(nrows),ncols_(ncols),mem_allocated(false),
    upper_(upper), trans_(trans), unitdiag_(unitdiag){
  if(trans){
    nrows_ = ncols;
    ncols_ = nrows;
  }
}

template<class T>
Matrix<T,TRI>::Matrix(const Matrix<T,TRI>& M)
  : nrows_(M.nrows_),ncols_(M.ncols_),arr_(M.arr_),mem_allocated(false),
    upper_(M.upper_),trans_(M.trans_),unitdiag_(M.unitdiag_){}

template<class T>
Matrix<T,TRI>::~Matrix(){
  if(mem_allocated)
    delete [] arr_;
}

template<class T>
void Matrix<T,TRI>::deep_copy(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(M.nrows(),M.ncols());
    *this = M;
  }
  upper_ = M.upper_;
  trans_ = M.trans_;
  unitdiag_ = M.unitdiag_;
}

template<class T>
void Matrix<T,TRI>::viewof(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(0,0);
    arr_ = M.arr_;
    mem_allocated = false;
  }
  nrows_ = M.nrows_;
  ncols_ = M.ncols_;
  upper_ = M.upper_;
  trans_ = M.trans_;
  unitdiag_ = M.unitdiag_;
}

template<class T>
void Matrix<T,TRI>::reallocate(size_t nrows,size_t ncols,T initval/* = 0*/){
  upper_ = true;
  trans_ = false;
  unitdiag_ = false;
  if( mem_allocated && (nrows_*ncols_ == nrows*ncols) ){
    nrows_ = nrows;
    ncols_ = ncols;
    SetAllTo(initval);
  }else if(nrows*ncols == 0){
    nrows_ = 0;
    ncols_ = 0;
    arr_ = 0;
    mem_allocated = false;
  }else{
    nrows_ = nrows;
    ncols_ = ncols;
    if(mem_allocated)
      delete [] arr_;
    arr_ = new T[nrows_ * ncols_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}

template<class T>
T& Matrix<T,TRI>::operator()(size_t i, size_t j){
  if(i>=nrows() || j>=ncols() )
    throw IndexError();
  else if(unitdiag_ && (i==j))
    throw MatVecError("cannot change diagonal of a unitdiag matrix");
  else if(upper_ && !trans_ && i<=j)
    return arr_[j*nrows_+i];
  else if(upper_ && trans_ && i>=j)
    return arr_[i*nrows_+j];
  else if(!upper_ && trans_ && i<=j)
    return arr_[i*nrows_+j];
  else if(upper_ && trans_ && i>=j)
    return arr_[i*nrows_+j];
  else
    throw MatVecError("cannot change undefined half of a triangular matrix");
}

template<class T>
const T Matrix<T,TRI>::operator()(size_t i,size_t j) const{
  if(i>=nrows() || j>=ncols() )
    throw IndexError();
  else if(unitdiag_ && (i==j)){
    return 1.0;
    //breaks some compilers:
    //return Matrix<T,TRI>::one;
  }
  else if(upper_ && !trans_ && i<=j)
    return arr_[j*nrows_+i];
  else if(upper_ && trans_ && i>=j)
    return arr_[i*nrows_+j];
  else if(!upper_ && trans_ && i<=j)
    return arr_[i*nrows_+j];
  else if(upper_ && trans_ && i>=j)
    return arr_[i*nrows_+j];
  else{
    return 0.0;
    //breaks some compilers:
    //return Matrix<T,TRI>::zero;
  }
}

template<class T>
Vector<T,GEN> Matrix<T,TRI>::diag(){
  if(unitdiag_)
    throw NotImplementedError("Matrix<T,TRI>::diag with unitdiag==true");
  else
    return Vector<T,GEN>( nrows_, arr_, nrows_+1);
}

template<class T>
const Vector<T,GEN> Matrix<T,TRI>::diag() const{
  if(unitdiag_)
    throw NotImplementedError("Matrix<T,TRI>::diag with unitdiag==true");
  else
    return Vector<T,GEN>( nrows_, arr_, nrows_+1);
}


template<class T>
void Matrix<T,TRI>::SetAllTo(T val){
  //call blas_COPY here?
  if(upper_){
    for(size_t i=0;i<nrows_;i++)
      for(size_t j=i;j<ncols_;j++)
	arr_[j*nrows_+i] = val;
  }else{
    for(size_t j=0;j<ncols_;j++)
      for(size_t i=j;i<nrows_;i++)
	arr_[j*nrows_+i] = val;
  }
}

template<class T>
T Matrix<T,TRI>::SumElements() const{
  T S = 0;
  size_t start = unitdiag_ ? 1 : 0;
  if(upper_){
    for(size_t i=0; i<nrows_; i++)
      for(size_t j=i+start; j<ncols_; j++)
	S += operator()(i,j);
  }else{
    for(size_t j=0; j<ncols_; j++)
      for(size_t i=j+start; i<nrows_; i++)
	S += operator()(i,j);
  }
  S += rank() * start;
  return S;
}


//explicitly declare double TRI matrices: 
// this is to prevent needing all the above
// template functions to be defined in the header
// (prevents code bloat...)
template class Matrix<double,TRI>;
