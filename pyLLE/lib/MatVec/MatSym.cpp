#include "MatSym.h"


/************************************************************
 * Matrix Definitions
 ************************************************************/

template<class T>
Matrix<T,SYM>::Matrix(size_t nrows/*=0*/,T initval/*=0*/)
  : nrows_(nrows),arr_(0),mem_allocated(false), upper_(true){
  if(nrows_ != 0){
    arr_ = new T[nrows_ * nrows_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}


template<class T>
Matrix<T,SYM>::Matrix(size_t nrows,T *arr,bool upper)
  : arr_(arr),nrows_(nrows),mem_allocated(false),upper_(upper){}

template<class T>
Matrix<T,SYM>::Matrix(const Matrix<T,SYM>& M)
  : nrows_(M.nrows_),arr_(M.arr_),
    mem_allocated(false),upper_(M.upper_){}

template<class T>
Matrix<T,SYM>::~Matrix(){
  if(mem_allocated)
    delete [] arr_;
}

template<class T>
void Matrix<T,SYM>::deep_copy(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(M.nrows());
    // call blas_COPY here?
    *this = M;
  }
  upper_ = M.upper_;
}

template<class T>
void Matrix<T,SYM>::viewof(const Matrix &M){
  if(arr_ != M.arr_){
    reallocate(0);
    arr_ = M.arr_;
    mem_allocated = false;
  }
  nrows_ = M.nrows_;
  upper_ = M.upper_;
}

template<class T>
void Matrix<T,SYM>::reallocate(size_t nrows,T initval/* = 0*/){
  upper_ = true;
  if( mem_allocated && (nrows_ == nrows) ){
    SetAllTo(initval);
  }else if(nrows == 0){
    nrows_ = 0;
    arr_ = 0;
    mem_allocated = false;
  }else{
    nrows_ = nrows;
    if(mem_allocated)
      delete [] arr_;
    arr_ = new T[nrows_ * nrows_];
    mem_allocated = true;
    SetAllTo(initval);
  }
}

template<class T>
T& Matrix<T,SYM>::operator()(size_t i, size_t j){
  if(i>=nrows_ || j>=nrows_)
    throw IndexError();
  else if( (upper_ && i<j) || (!upper_ && i>j) )
    return arr_[j*nrows_+i];
  else
    return arr_[i*nrows_+j];
}

template<class T>
const T& Matrix<T,SYM>::operator()(size_t i,size_t j) const{
  if(i>=nrows_ || j>=nrows_)
    throw IndexError();
  else if( (upper_ && i<j) || (!upper_ && i>j) )
    return arr_[j*nrows_+i];
  else
    return arr_[i*nrows_+j];
}

template<class T>
Vector<T,GEN> Matrix<T,SYM>::diag(){
  return Vector<T,GEN>( nrows_, arr_, nrows_+1);
}

template<class T>
const Vector<T,GEN> Matrix<T,SYM>::diag() const{
  return Vector<T,GEN>( nrows_, arr_, nrows_+1);
}


template<class T>
void Matrix<T,SYM>::SetAllTo(T val){
  //call blas_COPY here?
  for(size_t j=0;j<nrows_;j++)
    for(size_t i=j;i<nrows_;i++)
      operator()(i,j) = val;
}

template<class T>
T Matrix<T,SYM>::SumElements() const{
  T S = 0;
  if(upper_){
    for(size_t i=0; i<nrows_; i++)
      for(size_t j=i+1; j<nrows_; j++)
	S += operator()(i,j);
  }else{
    for(size_t j=0; j<nrows_; j++)
      for(size_t i=j+1; i<nrows_; i++)
	S += operator()(i,j);
  }
  S *= 2;
  for(size_t i=0;i<nrows_;i++)
    S+=operator()(i,i);
  return S;
}


//explicitly declare double SYM matrices: 
// this is to prevent needing all the above
// template functions to be defined in the header
// (prevents code bloat...)
template class Matrix<double,SYM>;
